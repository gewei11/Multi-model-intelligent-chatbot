import streamlit as st
import warnings
import os
import sys
import threading
import base64
from typing import Dict, Any
from PIL import Image
import io
import wave
import pyaudio
import nest_asyncio
import asyncio
import torch
import time
import torchaudio
import ChatTTS

# 禁用文件监视器警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置Streamlit配置
if not os.path.exists(".streamlit"):
    os.makedirs(".streamlit")
    
with open(".streamlit/config.toml", "w") as f:
    f.write("""
[server]
fileWatcherType = "none"

[global]
developmentMode = false
""")

# 设置页面 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="AI助手",
    page_icon="🤖",
    layout="wide"
)

# 改进的事件循环初始化
def init_event_loop():
    try:
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if not loop.is_running():
            def run_loop():
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                except Exception:
                    pass
            
            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            
        return loop
    except Exception as e:
        st.warning(f"事件循环初始化失败: {str(e)}")
        return None

# 初始化事件循环
nest_asyncio.apply()
loop = init_event_loop()

# 处理 PyTorch 兼容性
try:
    # 安全地访问和设置 PyTorch 配置
    if hasattr(torch, "_dynamo"):
        # 使用安全的方式访问和设置属性
        dynamo_config = getattr(torch, "_dynamo", None)
        if dynamo_config and hasattr(dynamo_config, "config"):
            config_obj = getattr(dynamo_config, "config", None)
            if config_obj and hasattr(config_obj, "disable_dynamic_shapes"):
                setattr(config_obj, "disable_dynamic_shapes", True)
    
    # 禁用梯度计算
    if hasattr(torch, "set_grad_enabled"):
        torch.set_grad_enabled(False)
    
    # 禁用JIT编译
    os.environ['PYTORCH_JIT'] = '0'
    
    # 设置确定性计算
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        # 禁用cudnn基准测试以提高稳定性
        if hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = False
except Exception as e:
    st.warning(f"PyTorch 配置警告: {str(e)}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.core_agent import CoreAgent
from utils.config import load_config
from utils.logger import setup_logger

# 设置界面样式
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stRadio [role=radiogroup] {
        gap: 1rem;
        padding: 0.5rem;
    }
    .stRadio label {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        min-width: 150px;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .uploaded-image {
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 加载配置和初始化
@st.cache_resource(show_spinner=True)
def init_agent():
    try:
        config = load_config()
        agent = CoreAgent(config)
        return agent
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        return None

# 添加错误处理
agent = init_agent()
if agent is None:
    st.error("系统初始化失败，请检查配置并重试。")
    st.stop()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 检查必要的依赖
def check_dependencies():
    missing_deps = []
    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio")
    
    if missing_deps:
        st.error(f"缺少必要的依赖: {', '.join(missing_deps)}\n\n"
                 f"请运行以下命令安装：\n"
                 f"Windows: pip install pipwin && pipwin install pyaudio\n"
                 f"其他系统: pip install pyaudio")
        return False
    return True

# 侧边栏设置 - 使用容器美化布局
with st.sidebar:
    st.title("⚙️ 设置")
    with st.container():
        st.subheader("模型设置")
        model_option = st.radio(
            label="选择模型",  # 添加有意义的标签
            options=["自动（智能选择）", "Qwen2.5", "DeepSeek", "混合模式"],
            label_visibility="visible"  # 明确设置标签可见性
        )
    
    st.divider()
    
    with st.container():
        st.subheader("功能设置")
        input_mode = st.radio(
            label="选择输入模式",  # 添加有意义的标签
            options=["文本输入", "图片分析", "语音对话"],
            key="input_mode",
            label_visibility="visible"  # 明确设置标签可见性
        )
        st.divider()  # 添加分隔线
        sentiment_enabled = st.toggle("情感分析", value=True)
        show_analysis = st.toggle("显示分析结果", value=True) if sentiment_enabled else False
        weather_enabled = st.toggle("天气查询", value=True)
    
    if st.button("清除对话历史", type="primary"):
        st.session_state.messages = []
        agent.clear_history()
        st.experimental_rerun()

# 主界面
st.title("🤖 AI助手")

# 显示聊天历史
st.container()  # 创建容器用于显示消息历史
for i, message in enumerate(st.session_state.messages):
    # 检查是否是重复消息
    if i > 0 and message["content"] == st.session_state.messages[i-1]["content"]:
        continue  # 跳过重复消息
        
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 根据选择的输入模式显示对应界面
if input_mode == "语音对话":
    try:
        if check_dependencies():
            # 添加文件上传组件
            uploaded_audio = st.file_uploader(
                label="上传音频文件",
                type=["wav"],
                help="支持WAV格式的音频文件"
            )
            
            # 添加录音时长选择
            record_duration = st.radio(
                "选择录音时长",
                [5, 10, 15],
                format_func=lambda x: f"{x}秒",
                horizontal=True
            )
            
            # 初始化会话状态
            if "recording" not in st.session_state:
                st.session_state.recording = False
                st.session_state.audio_frames = []
                st.session_state.pyaudio_instance = None
                st.session_state.stream = None
                st.session_state.recording_start_time = None
            
            if st.button(f"🎤 开始录音 (将录制{record_duration}秒)", 
                        key="start_record",
                        disabled=st.session_state.recording):
                try:
                    # 开始录音
                    CHUNK = 1024
                    FORMAT = pyaudio.paInt16
                    CHANNELS = 1
                    RATE = 16000
                    
                    p = pyaudio.PyAudio()
                    stream = p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
                    
                    st.session_state.recording = True
                    st.session_state.audio_frames = []
                    st.session_state.pyaudio_instance = p
                    st.session_state.stream = stream
                    st.session_state.recording_start_time = time.time()
                    
                    # 显示录音进度
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # 录音循环使用选择的时长
                    for i in range(0, int(RATE / CHUNK * record_duration)):
                        if st.session_state.stream:
                            data = st.session_state.stream.read(CHUNK, exception_on_overflow=False)
                            st.session_state.audio_frames.append(data)
                            
                            # 更新进度
                            elapsed_time = time.time() - st.session_state.recording_start_time
                            progress = min(1.0, elapsed_time / record_duration)
                            remaining_time = max(0, record_duration - elapsed_time)
                            progress_text.text(f"正在录音... 还剩 {remaining_time:.1f} 秒")
                            progress_bar.progress(progress)
                    
                    # 录音完成后自动处理
                    if st.session_state.recording:
                        progress_text.text("录音完成，正在处理...")
                        
                        # 保存录音
                        temp_file = os.path.join("temp", "temp_recording.wav")
                        os.makedirs("temp", exist_ok=True)
                        
                        with wave.open(temp_file, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(st.session_state.audio_frames))
                        
                        # 关闭音频流
                        if st.session_state.stream:
                            st.session_state.stream.stop_stream()
                            st.session_state.stream.close()
                        if st.session_state.pyaudio_instance:
                            st.session_state.pyaudio_instance.terminate()
                        
                        # 识别音频
                        with st.spinner("正在识别语音..."):
                            recognized_text = agent.voice_agent._recognize_from_file(temp_file)
                            
                            if recognized_text:
                                st.success(f"识别结果: {recognized_text}")
                                # 添加用户消息到历史
                                st.session_state.messages.append({"role": "user", "content": recognized_text})
                                with st.chat_message("user"):
                                    st.markdown(recognized_text)
                                
                                # 生成助手回复
                                with st.chat_message("assistant"):
                                    message_placeholder = st.empty()
                                    full_response = ""
                                    
                                    # 处理识别结果
                                    for chunk in agent.process_input(
                                        recognized_text,
                                        {
                                            "model_option": model_option,
                                            "sentiment_enabled": sentiment_enabled,
                                            "show_analysis": show_analysis,
                                            "stream": True
                                        }
                                    ):
                                        full_response += chunk
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    message_placeholder.markdown(full_response)
                                    # 添加助手回复到历史
                                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                                    
                                    # 将文本转换为语音输出
                                    with st.spinner("正在生成语音..."):
                                        try:
                                            # 初始化ChatTTS
                                            chat = ChatTTS.Chat()
                                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                            chat.load(source='huggingface', device=device)
                                            
                                            # 采样说话人和设置参数
                                            rand_spk = chat.sample_random_speaker()
                                            params_infer_code = ChatTTS.Chat.InferCodeParams(
                                                spk_emb=rand_spk,
                                                temperature=0.3,
                                                top_P=0.7,
                                                top_K=20,
                                            )
                                            
                                            params_refine_text = ChatTTS.Chat.RefineTextParams(
                                                prompt='[oral_2][laugh_0][break_6]',
                                            )
                                            
                                            # 确保输出目录存在
                                            output_dir = "temp/tts_output"
                                            os.makedirs(output_dir, exist_ok=True)
                                            
                                            # 生成语音
                                            wavs = chat.infer(
                                                [full_response],
                                                params_refine_text=params_refine_text,
                                                params_infer_code=params_infer_code,
                                            )
                                            
                                            # 保存音频文件
                                            output_file = f"{output_dir}/response.wav"
                                            try:
                                                torchaudio.save(output_file, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
                                            except:
                                                torchaudio.save(output_file, torch.from_numpy(wavs[0]), 24000)
                                            
                                            # 读取生成的音频文件
                                            if os.path.exists(output_file):
                                                audio_file = open(output_file, 'rb')
                                                audio_bytes = audio_file.read()
                                                audio_file.close()
                                                
                                                # 在界面上显示音频播放器
                                                st.audio(audio_bytes, format='audio/wav')
                                                
                                                # 清理临时文件
                                                try:
                                                    os.remove(output_file)
                                                except:
                                                    pass
                                            else:
                                                st.warning("语音文件生成失败")
                                                
                                        except Exception as e:
                                            st.error(f"语音合成失败: {str(e)}")

                            else:
                                st.warning("未能识别出语音内容，请重新录音")
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                        
                        # 重置录音状态
                        st.session_state.recording = False
                        st.session_state.audio_frames = []
                        st.session_state.stream = None
                        st.session_state.pyaudio_instance = None
                        st.session_state.recording_start_time = None
                        
                except Exception as e:
                    st.error(f"录音过程出错: {str(e)}")
                    # 确保资源被释放
                    if hasattr(st.session_state, 'stream') and st.session_state.stream:
                        st.session_state.stream.stop_stream()
                        st.session_state.stream.close()
                    if hasattr(st.session_state, 'pyaudio_instance') and st.session_state.pyaudio_instance:
                        st.session_state.pyaudio_instance.terminate()
                    
                    # 重置状态
                    st.session_state.recording = False
                    st.session_state.audio_frames = []
                    st.session_state.stream = None
                    st.session_state.pyaudio_instance = None
                    st.session_state.recording_start_time = None

            # 处理上传的音频文件
            if uploaded_audio is not None:
                with st.spinner("处理语音中..."):
                    try:
                        result = agent.voice_agent.process_input(uploaded_audio.getvalue())
                        if "error" not in result:
                            st.success(f"识别结果: {result['recognized_text']}")
                        else:
                            st.error(result["error"])
                    except Exception as e:
                        st.error(f"处理语音失败: {str(e)}")

        else:
            st.info("请安装所需依赖后继续使用语音功能")

    except Exception as e:
        st.error(f"语音功能出错: {str(e)}")

elif input_mode == "文本输入":
    # 文本输入模式
    if prompt := st.chat_input("请输入您的问题...", key="text_input"):
        # 避免重复添加相同的消息
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 生成助手回复
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # 添加加载动画
                    with st.spinner('思考中...'):
                        # 创建一个空的消息占位符
                        message_placeholder = st.empty()
                        full_response = ""
                        buffer = ""
                        last_update_time = time.time()
                        update_interval = 0.1  # 控制更新频率，每0.1秒更新一次
                        
                        for response_chunk in agent.process_input(prompt, {
                            "model_option": model_option,
                            "weather_enabled": weather_enabled,
                            "sentiment_enabled": sentiment_enabled,
                            "show_analysis": show_analysis,
                            "stream": True
                        }):
                            buffer += response_chunk
                            current_time = time.time()
                            
                            # 当积累的文本超过一定长度或达到更新间隔时更新显示
                            if len(buffer) >= 5 or (current_time - last_update_time) >= update_interval:
                                full_response += buffer
                                message_placeholder.markdown(full_response + "▌")
                                buffer = ""
                                last_update_time = current_time
                                time.sleep(0.02)  # 添加小延迟以实现打字机效果
                    
                    # 确保显示最后的文本
                    if buffer:
                        full_response += buffer
                        message_placeholder.markdown(full_response)
                    
                    # 避免重复添加相同的回复
                    if not st.session_state.messages or st.session_state.messages[-1]["content"] != full_response:
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = f"⚠️ 生成回复时发生错误: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif input_mode == "图片分析":
    # 使用columns布局美化上传区域
    col1, col2, _ = st.columns([2, 2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png"],
            help="支持jpg、jpeg、png格式的图片"
        )
    
    if uploaded_file is not None:
        # 显示上传的图片,添加样式
        image = Image.open(uploaded_file)
        st.image(
            image, 
            caption="上传的图片", 
            use_container_width=True  # 使用新参数替换use_column_width
        )
        
        # 美化提示输入区域
        with st.container():
            prompt = st.text_input(
                "请输入关于图片的问题",
                value="这张图片里有什么？",
                placeholder="例如:描述一下这张图片的内容..."
            )
        
        # 美化分析按钮
        if st.button("✨ 开始分析", key="analyze_btn"):
            try:
                # 添加进度条和加载动画
                with st.spinner('正在分析图片...'):
                    progress_bar = st.progress(0)
                    
                    # 保存图片处理
                    temp_image_path = os.path.join("temp", "temp_image.png")
                    os.makedirs("temp", exist_ok=True)
                    image.save(temp_image_path)
                    progress_bar.progress(30)

                    # 添加到对话历史
                    st.session_state.messages.append({"role": "user", "content": f"[图片分析] {prompt}"})
                    progress_bar.progress(50)
                    
                    # 显示用户输入
                    with st.chat_message("user"):
                        st.markdown(f"🖼️ [图片] {prompt}")
                    
                    # 调用模型分析
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        config = {
                            "models": {
                                "minicpm": {
                                    "model_name": "aiden_lu/minicpm-v2.6:Q4_K_M",
                                    "api_base": "http://localhost:11434/v1/chat/completions",
                                    "temperature": 0.7,
                                    "max_tokens": 2048,
                                    "vision": True
                                }
                            }
                        }
                        
                        from models.minicpm import MiniCPMModel
                        model = MiniCPMModel(config["models"]["minicpm"])
                        progress_bar.progress(70)
                        
                        for chunk in model.generate(prompt, images=[temp_image_path]):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        progress_bar.progress(100)
                    
                    # 清理临时文件
                    os.remove(temp_image_path)
                    
            except Exception as e:
                error_msg = f"⚠️ 图片分析失败: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})