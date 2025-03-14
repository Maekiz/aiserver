import jwt
from dotenv import load_dotenv
import os

load_dotenv()

auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyRGF0YSI6eyJpZCI6ImIxYzRiZmFjLTRkM2UtNGMyZi05OTQ3LWIxMGYxZTA2YjkxNSIsImVtYWlsIjoibWFpbEBhbGVrc2FuZGVyZWttYW4ubm8iLCJ2ZXJpZmllZF9lbWFpbCI6dHJ1ZSwidXNlcm5hbWUiOiJhbGVrc2FuZGVyZWttYW4ifSwiaWF0IjoxNzQxOTc0NDMxLCJleHAiOjE3NDE5NzU2MzF9.ZB3FLDsCCE3y8wpMLVasEnQQ-GIuHaA3HFipvbLgsCo"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

try:
    decoded_data = jwt.decode(
        auth_token,
        JWT_SECRET_KEY,
        algorithms=['HS256'],
        options={"verify_exp": True} 
    )
    username = decoded_data['userData']['username']
except:
    print("Invalid token")