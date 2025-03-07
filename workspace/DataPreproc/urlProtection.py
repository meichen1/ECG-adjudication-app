from itsdangerous import URLSafeSerializer
import os

secret_key = os.getenv('SECRET_KEY_FOR_TOKEN') 
if not secret_key:
    raise ValueError('SECRET_KEY_FOR_TOKEN not found in environment variables')
serializer = URLSafeSerializer(secret_key)

# Generate two tokens
token1 = serializer.dumps({'user': 'physician1'})
token2 = serializer.dumps({'user': 'physician2'})

print(f'Token 1: {token1}')
print(f'Token 2: {token2}')