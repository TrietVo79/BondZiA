#!/bin/bash
git pull origin main
git add .
git commit -m "Auto-sync: Cập nhật thay đổi" || true
git push origin main