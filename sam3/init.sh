#!/bin/bash

# 1. 연결할 대상(Target) 경로 설정 (부모 폴더 밑의 tools)
TARGET="../tools"
LINK_NAME="tools"

# 2. 부모 폴더에 실제로 tools가 있는지 확인
if [ -d "$TARGET" ]; then
    # 3. 이미 링크나 폴더가 존재하는지 확인
    if [ -e "$LINK_NAME" ]; then
        echo "⚠️  현재 폴더에 이미 '$LINK_NAME' 파일/폴더가 존재합니다."
    else
        # 4. 심볼릭 링크 생성 (핵심 명령어)
        ln -s "$TARGET" "$LINK_NAME"
        echo "✅ 성공: 부모 폴더의 tools를 현재 폴더로 링크했습니다."
    fi
else
    echo "❌ 에러: 부모 폴더('../')에서 'tools' 디렉토리를 찾을 수 없습니다."
fi

ln -s ./blobs/3e600e765b0f1b4c52680aed9ad242c9a90f7ed1 config.json
ln -s ./blobs/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e sam3.pt