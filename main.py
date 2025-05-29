import os
import sys
import cv2
import numpy as np
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime
import boto3
from botocore.client import Config
import qrcode
import random

###test purposes    
import base64
from typing import NamedTuple

class Entry(NamedTuple):
    b64_json: str

CAMERA_TRANS_LEVEL = 0.9
CAMERA_INDEX=1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_PATH = os.path.join(BASE_DIR, "static", "mainframe.png")

prompt_list = ["Turn this photo into a Van-gogh style Recognizable as me",
"Turn this photo into a Pixel-art avatar Recognizable as me",
"Turn this photo into a Simpsons character Recognizable as me"]

class PhotoStylizer:
    def __init__(self,
                 camera_index=CAMERA_INDEX,
                 silhouette_path="frame.png",
                 trans_lvl=CAMERA_TRANS_LEVEL,
                 upload_dir: str = "static/uploads",
                 stylized_dir: str = "static/stylized",
                 qr_dir: str = "static/qr"):
        self.upload_dir = upload_dir
        self.stylized_dir = stylized_dir
        self.qr_dir = qr_dir
        # ensure dirs exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.stylized_dir, exist_ok=True)
        os.makedirs(self.qr_dir, exist_ok=True)
        # Load environment
        load_dotenv()
        # OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment")
        #S3 client
        self._init_s3_client()
        self.camera_index = camera_index
        self.silhouette_path = silhouette_path
        self.trans_lvl = trans_lvl

    def _init_s3_client(self):
        # Load AWS creds & bucket from .env
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = "eu-north-1"
        bucket = os.getenv("AWS_BUCKET")
        if not all([key_id, secret, region, bucket]):
            raise RuntimeError("Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                               "AWS_DEFAULT_REGION & AWS_BUCKET in .env")
        # Standard AWS S3 client
        self.s3 = boto3.client(
            "s3",
            region_name="eu-north-1",
            endpoint_url=f"https://s3.{region}.amazonaws.com",   # ← force regional host
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4"),
        )
        self.s3_bucket = bucket

    def camera_mask(self, frame):
        sil = cv2.imread(self.silhouette_path, cv2.IMREAD_UNCHANGED)
        if sil is None:
            raise FileNotFoundError(f"{self.silhouette_path} not found.")

        sil_resized = cv2.resize(
            sil, (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA
        )
        b, g, r, a = cv2.split(sil_resized)
        alpha = (a.astype(float) / 255.0) * self.trans_lvl

        overlay_rgb = cv2.merge((b, g, r))
        frame_float = frame.astype(float)
        
        for c in range(3):
            frame_float[:, :, c] = (
                alpha * overlay_rgb[:, :, c] +
                (1 - alpha) * frame_float[:, :, c]
            )

        return frame_float.astype(np.uint8)

    def capture_photo(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        print("🔲 SPACE to capture, ESC to exit")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            blended = self.camera_mask(frame)
            cv2.imshow("Live", blended)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                cv2.imwrite("photo.png", blended)
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
        cap.release()
        cv2.destroyAllWindows()
        return frame

    def crop_and_resize(self, frame):
        h, w = frame.shape[:2]
        m = min(h, w)
        crop = frame[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite("photo_256.png", resized)
        return resized

    def create_mask(self):
        # opaque mask
        mask = 255 * np.ones((256, 256), dtype=np.uint8)
        cv2.imwrite("mask_256.png", mask)

    def call_edit_api(self, prompt, model="gpt-image-1", size="1024x1024"):
        resp = self.client.images.edit(
            model=model,
            image=open("photo_256.png", "rb"),
            mask=open("mask_256.png", "rb"),
            prompt=prompt,
            n=1,
            size=size
        )
        return resp.data[0]
        #with open("photo_256.png", "rb") as img_file:
        #    raw_bytes = img_file.read()
        #b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        #return Entry(b64_json=b64_str)
    
    def propmt_randomizer(self, p_list=prompt_list):
        return random.sample(p_list, 1)[0]

    @staticmethod
    def decode_entry(entry):
        if entry.b64_json:
            return base64.b64decode(entry.b64_json)
        return requests.get(entry.url).content

    @staticmethod
    def hour_minute():
        now = datetime.now()
        return f"{now.hour}-{now.minute}"

    def save_output(self, img_bytes, email=""):
        timestamp = self.hour_minute()
        fname = f"{email}_stylized_photo_{timestamp}.png"
        out_path = os.path.join(self.upload_dir, fname)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ Saved locally: {out_path}")
        return out_path
    

    def frame_photo(self, photo_path, frame_path=FRAME_PATH,  position=(32, 306)):
        frame = Image.open(frame_path).convert("RGBA")
        photo = Image.open(photo_path).convert("RGBA")
        x, y = position
        # Paste & save
        frame.paste(photo, (x, y), photo)
        base, ext = os.path.splitext(os.path.basename(photo_path))
        out_name = f"{base}_framed{ext}"
        out_path = os.path.join(self.stylized_dir, out_name)
        frame.save(out_path)
        print(f"✅ Framed image saved: {out_path}")
        return out_path
    

    def create_qr(self, presigned_url, out_path):
        # inside PhotoStylizer.run(), after upload_to_s3:
        qr = qrcode.make(presigned_url)
        qr_fname = f"qr_{os.path.basename(out_path)}"
        qr_path = os.path.join(self.qr_dir, qr_fname)
        qr.save(qr_path)
        print("CHECK::::", qr_path)
        return qr_fname

    def upload_to_s3(self, local_path, key=None):
        """Uploads and returns a 24h presigned URL."""
        if key is None:
            key = os.path.basename(local_path)
        # 1) Upload
        self.s3.upload_file(local_path, self.s3_bucket, key)
        # 2) Presign for 24h
        url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.s3_bucket, "Key": key},
            ExpiresIn=86400,
        )
        print(f"✅ Uploaded & presigned URL: {url}")
        return url

    def run(self, frame, email):
        #frame = self.capture_photo()
        self.crop_and_resize(frame)
        self.create_mask()

        # you can swap prompt here
        prompt = "Turn this photo into a cyberpunk2077 style Recognizable as me" #self.propmt_randomizer()
        entry = self.call_edit_api(prompt)

        img_bytes = self.decode_entry(entry)
        out_path = self.save_output(img_bytes, email)
        framed_path = self.frame_photo(out_path)
        # finally, upload to Cloudflare R2
        url = self.upload_to_s3(framed_path)

        qr_path = self.create_qr(url, out_path)
        return url, out_path, qr_path 
        

if __name__ == "__main__":
    PhotoStylizer().run()
