#!/usr/bin/env python3
import os
import sys
import stat

PATHS_TO_CHECK = [
    "/opt/vllm-env",
    "/opt/vllm-env/bin",
    "/opt/vllm-env/bin/vllm",
    "/opt/vllm-env/version",
    "/opt/diffusers-env/bin/python3",
    "/opt/sglang-env/bin/python3"
]

def check_path(path):
    print(f"Checking: {path}")
    try:
        st = os.stat(path)
        print(f"  ✅ Exists")
        print(f"  Mode: {stat.filemode(st.st_mode)}")
        print(f"  UID: {st.st_uid}, GID: {st.st_gid}")
        
        # Check lstat to see if it is a link
        lst = os.lstat(path)
        if stat.S_ISLNK(lst.st_mode):
             target = os.readlink(path)
             print(f"  🔗 Symlink to: {target}")
             if not os.path.exists(target):
                 print(f"  ❌ TARGET DOES NOT EXIST!")
             elif not os.access(target, os.R_OK):
                 print(f"  ⚠️  TARGET NOT READABLE (permissions?)")

    except FileNotFoundError:
        print(f"  ❌ Not Found")
    except PermissionError:
        print(f"  ❌ Permission Denied")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print("-" * 40)

def main():
    print(f"User ID: {os.getuid()}")
    print("-" * 40)
    for p in PATHS_TO_CHECK:
        check_path(p)

if __name__ == "__main__":
    main()
