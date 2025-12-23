## BRICK Agent - 使用指南

### 步骤 1：克隆项目仓库

```bash
git clone -b master https://github.com/lyt2003-yt/Biomics_agent.git
```

### 步骤 2：配置环境变量

编辑 `graph/brick_test_config.env` 文件，修改以下配置：

```bash
# 修改项目根路径为实际路径
PROJECT_ROOT=/your/path/to/Biomics_agent
```

### 步骤 3：创建 Conda 环境

使用提供的环境配置文件创建 Python 环境：

```bash
# 使用 conda 创建环境
conda env create -f biomics_environment.yml

# 激活环境
conda activate biomics_agent
```

### 步骤 4：启动 Docker 沙箱容器

#### 4.1 下载 Docker 镜像

从阿里云容器镜像服务下载预构建的镜像：

```bash
# 下载镜像（约 6.57GB）
docker pull crpi-b88i7r04wqgzpar4.cn-beijing.personal.cr.aliyuncs.com/biomics/biomics-agent:v6

# 可选：给镜像打一个标签
docker tag crpi-b88i7r04wqgzpar4.cn-beijing.personal.cr.aliyuncs.com/biomics/biomics-agent:v6 biomics_agent:v6
```

#### 4.2 启动沙箱容器

```bash
# 在项目目录下新建一个data文件：
Biomics_agent/data

# 停止并删除旧容器（如果存在）
docker rm -f my_code_sandbox

# 启动新的沙箱容器
docker run -d \
  --name my_code_sandbox \
  --network="host" \
  -e PYTHONUNBUFFERED=1 \
  -v /your/path/to/Biomics_agent/data:/workspace/data \
  biomics_agent:v6 \
  jupyter kernelgateway \
  --KernelGatewayApp.ip=0.0.0.0 \
  --KernelGatewayApp.port=8888 \
  --KernelGatewayApp.auth_token="" \
  --JupyterWebsocketPersonality.list_kernels=True \
  --KernelManager.ip=0.0.0.0
```

**说明：**
- 将 `/your/path/to/Biomics_agent/data` 换成项目 `data` 目录的**绝对路径**
- 容器会在 `8888` 端口启动 Jupyter Kernel Gateway 服务
- 使用 `--network="host"` 使容器与主机共享网络

**验证容器运行：**
```bash
# 检查容器状态
docker ps | grep my_code_sandbox

# 测试连接
curl http://127.0.0.1:8888/api
```

### 步骤 5：运行应用

启动 NiceGUI Web 应用：

```bash
# 确保在项目根目录
cd /your/path/to/Biomics_agent

# 激活环境（如果还没激活）
conda activate biomics_agent

# 启动应用
python app_nicegui.py
```

应用启动后，浏览器会自动打开或访问：
```
http://localhost:8080
```

