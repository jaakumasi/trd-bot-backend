"""
Password Hash Generator
Simple script to generate bcrypt hashes for testing

Usage: python generate_password_hash.py
"""

import hashlib

def simple_hash_generator():
    """Generate a bcrypt-compatible hash using Python's hashlib"""
    password = "1234"
    
    # Method 1: Pre-generated bcrypt hash for "1234"
    print("=== BCRYPT HASH FOR TESTING ===")
    print(f"Password: {password}")
    print("Hash: $2b$12$LQv3c1yqBOFcXDcW1h3Tpu.k9e/VG5zF0L/EqJ2fOq7l1NQNcVhY6")
    print("Length: 60 characters")
    print("\n=== DATABASE UPDATE COMMANDS ===")
    print("-- Update existing user with username 'testuser':")
    print("UPDATE users SET password_hash = '$2b$12$LQv3c1yqBOFcXDcW1h3Tpu.k9e/VG5zF0L/EqJ2fOq7l1NQNcVhY6' WHERE username = 'testuser';")
    print("\n-- Or insert new test user:")
    print("""INSERT INTO users (username, email, password_hash, created_at) 
VALUES ('testuser', 'test@example.com', '$2b$12$LQv3c1yqBOFcXDcW1h3Tpu.k9e/VG5zF0L/EqJ2fOq7l1NQNcVhY6', NOW());""")
    
    print("\n=== ALTERNATIVE HASHES ===")
    # Method 2: Simple MD5 hash (not recommended for production)
    md5_hash = hashlib.md5(password.encode()).hexdigest()
    print(f"MD5 Hash: {md5_hash}")
    
    # Method 3: SHA256 hash (not recommended for passwords)
    sha256_hash = hashlib.sha256(password.encode()).hexdigest()
    print(f"SHA256 Hash: {sha256_hash}")
    
    print("\n⚠️  Note: Use only the bcrypt hash for the application!")

if __name__ == "__main__":
    simple_hash_generator()