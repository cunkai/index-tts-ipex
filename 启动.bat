@echo off
REM 激活 Anaconda 环境并运行 webui.py

REM 切换到 Anaconda 的脚本目录
call "D:\Apps\Anaconda3PythonManage\Scripts\activate.bat"

REM 激活你的虚拟环境
call conda activate index-tts-bilibili

REM 切换到 webui.py 所在目录
cd /d D:\Apps\index-tts\

REM 设置启动参数
set HOST=127.0.0.1
set PORT=7860
set MODEL_DIR=checkpoints

REM 启动 WebUI（后台启动）
start "" python webui.py --host %HOST% --port %PORT% --model_dir %MODEL_DIR%

REM 防止窗口运行完后自动关闭
pause
