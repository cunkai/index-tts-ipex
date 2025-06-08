@echo off
chcp 65001 >nul

REM === 主程序 ===
REM 获取当前脚本所在目录（项目根目录）
set "ROOT_DIR=%~dp0"
set "PYTHON=%ROOT_DIR%python\python.exe"

REM 设置启动参数
set HOST=127.0.0.1
set PORT=7860
set MODEL_DIR=checkpoints

REM 切换到项目根目录
cd /d "%ROOT_DIR%"

REM 使用项目中的Python解释器运行WebUI
echo 正在启动WebUI服务...
echo 模型目录: %MODEL_DIR%
echo.

%PYTHON% webui.py --host %HOST% --port %PORT% --model_dir %MODEL_DIR%

REM 服务停止后提示
echo.
echo [WebUI 服务已停止，按任意键退出...]
pause >nul