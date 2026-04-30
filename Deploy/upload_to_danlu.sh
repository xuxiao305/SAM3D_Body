#!/bin/bash
# ============================================================
# SAM3D_Body 本地 → 丹炉 代码上传脚本
# 在 WSL2 环境下执行
# ============================================================

set -e

# ---- 配置 ----
SSH_KEY="/tmp/DanLu_key"
SSH_HOST="root@apps-sl.danlu.netease.com"
SSH_PORT="44304"
SSH_OPTS="-i ${SSH_KEY} -p ${SSH_PORT} -o StrictHostKeyChecking=no"

LOCAL_MAIN="/mnt/d/AI/Prototypes/SAM3D_Body/Main/"
LOCAL_DEPLOY="/mnt/d/AI/Prototypes/SAM3D_Body/Deploy/"
REMOTE_DIR="/root/SAM3D_Body"

# ---- 准备密钥 ----
echo "准备 SSH 密钥 ..."
cp /mnt/d/AI/PrivateKeys/DanLu/xuxiao02_rsa ${SSH_KEY}
chmod 600 ${SSH_KEY}

# ---- 上传项目代码 ----
echo ""
echo "上传项目代码 Main/ → ${REMOTE_DIR}/ ..."
rsync -aL --info=progress2 \
    -e "ssh ${SSH_OPTS}" \
    ${LOCAL_MAIN} \
    ${SSH_HOST}:${REMOTE_DIR}/

# ---- 上传部署脚本 ----
echo ""
echo "上传部署脚本 Deploy/ → ${REMOTE_DIR}/Deploy/ ..."
rsync -aL --info=progress2 \
    -e "ssh ${SSH_OPTS}" \
    ${LOCAL_DEPLOY} \
    ${SSH_HOST}:${REMOTE_DIR}/Deploy/

# ---- 在服务器上设置脚本权限 ----
echo ""
echo "设置脚本执行权限 ..."
ssh ${SSH_OPTS} ${SSH_HOST} "chmod +x ${REMOTE_DIR}/Deploy/*.sh"

echo ""
echo "✅ 上传完成！"
echo ""
echo " 下一步："
echo "   1. SSH 登录服务器: ssh ${SSH_OPTS} ${SSH_HOST}"
echo "   2. 运行部署脚本: bash ${REMOTE_DIR}/Deploy/setup_danlu.sh"
echo "   3. 下载模型: bash ${REMOTE_DIR}/Deploy/download_models.sh"
