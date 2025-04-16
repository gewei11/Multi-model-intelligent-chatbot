import os
import json
from typing import Dict, Any, List, Optional, Union, BinaryIO
import numpy as np
from PIL import Image
import cv2

from utils.logger import get_logger
from utils.helper_functions import retry

class ImageAgent:
    """
    图像处理Agent，负责处理用户上传的图片，进行图像识别和分析
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图像处理Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("image_agent")
        self.logger.info("图像处理Agent初始化")
        
        # 输出目录
        self.output_dir = config.get("output_dir", "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 支持的图像处理功能
        self.supported_features = [
            "图像识别",
            "物体检测",
            "文字识别",
            "图像分类",
            "图像增强",
            "人脸检测"
        ]
    
    def process(self, image_data: Union[str, bytes, BinaryIO, np.ndarray], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理图像数据，进行识别和分析
        
        Args:
            image_data: 图像数据，可以是文件路径、字节数据、文件对象或numpy数组
            context: 上下文信息，包含处理要求等
            
        Returns:
            处理结果字典
        """
        if context is None:
            context = {}
        
        self.logger.info("开始处理图像数据")
        
        try:
            # 加载图像
            image = self._load_image(image_data)
            
            # 根据上下文确定处理类型
            process_type = context.get("process_type", "general")
            
            # 根据处理类型调用相应的处理函数
            if process_type == "object_detection":
                result = self._detect_objects(image)
            elif process_type == "text_recognition":
                result = self._recognize_text(image)
            elif process_type == "image_classification":
                result = self._classify_image(image)
            elif process_type == "image_enhancement":
                result = self._enhance_image(image)
            elif process_type == "face_detection":
                result = self._detect_faces(image)
            else:
                # 默认进行通用图像分析
                result = self._analyze_image(image)
            
            self.logger.info(f"图像处理完成: {result['summary']}")
            return result
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": "图像处理失败，请检查图像格式或尝试上传其他图片。"
            }
    
    def _load_image(self, image_data: Union[str, bytes, BinaryIO, np.ndarray]) -> np.ndarray:
        """
        加载图像数据
        
        Args:
            image_data: 图像数据，可以是文件路径、字节数据、文件对象或numpy数组
            
        Returns:
            numpy数组格式的图像
        """
        try:
            if isinstance(image_data, str):
                # 输入是文件路径
                if not os.path.exists(image_data):
                    raise FileNotFoundError(f"图像文件不存在: {image_data}")
                image = cv2.imread(image_data)
                if image is None:
                    raise ValueError(f"无法读取图像文件: {image_data}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                
            elif isinstance(image_data, bytes):
                # 输入是字节数据
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("无法解码图像数据")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                
            elif isinstance(image_data, np.ndarray):
                # 输入已经是numpy数组
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    return image_data
                else:
                    raise ValueError("图像数组格式不正确，应为RGB格式")
                    
            else:
                # 尝试作为文件对象读取
                image_data.seek(0)  # 确保从文件开头读取
                image_bytes = image_data.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("无法从文件对象读取图像")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                
        except Exception as e:
            self.logger.error(f"加载图像失败: {str(e)}")
            raise
    
    def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        通用图像分析
        
        Args:
            image: 图像数据
            
        Returns:
            分析结果
        """
        # 获取图像基本信息
        height, width, channels = image.shape
        
        # 计算图像统计信息
        brightness = np.mean(image)
        contrast = np.std(image)
        
        # 颜色分布
        color_distribution = {
            "red": float(np.mean(image[:, :, 0])),
            "green": float(np.mean(image[:, :, 1])),
            "blue": float(np.mean(image[:, :, 2]))
        }
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "analyzed_image.jpg")
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        
        return {
            "status": "success",
            "summary": "图像分析完成",
            "image_info": {
                "width": width,
                "height": height,
                "channels": channels,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "color_distribution": color_distribution
            },
            "output_path": output_path
        }
    
    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        物体检测
        
        Args:
            image: 图像数据
            
        Returns:
            检测结果
        """
        # 这里应该使用物体检测模型，如YOLO、SSD等
        # 为简化示例，这里使用模拟数据
        
        # 模拟检测结果
        detected_objects = [
            {"label": "人", "confidence": 0.92, "bbox": [100, 150, 200, 300]},
            {"label": "椅子", "confidence": 0.85, "bbox": [300, 200, 100, 150]},
            {"label": "桌子", "confidence": 0.78, "bbox": [250, 300, 200, 100]}
        ]
        
        # 在图像上标记检测结果
        result_image = image.copy()
        for obj in detected_objects:
            x, y, w, h = obj["bbox"]
            label = f"{obj['label']} {obj['confidence']:.2f}"
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "detected_objects.jpg")
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_rgb)
        
        return {
            "status": "success",
            "summary": f"检测到{len(detected_objects)}个物体",
            "objects": detected_objects,
            "output_path": output_path
        }
    
    def _recognize_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        文字识别
        
        Args:
            image: 图像数据
            
        Returns:
            识别结果
        """
        # 这里应该使用OCR模型，如Tesseract、PaddleOCR等
        # 为简化示例，这里使用模拟数据
        
        # 模拟识别结果
        recognized_text = "这是一段从图像中识别出的示例文字，实际应用中应该使用OCR引擎进行识别。"
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "recognized_text.jpg")
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        
        return {
            "status": "success",
            "summary": "文字识别完成",
            "text": recognized_text,
            "output_path": output_path
        }
    
    def _classify_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        图像分类
        
        Args:
            image: 图像数据
            
        Returns:
            分类结果
        """
        # 这里应该使用图像分类模型，如ResNet、MobileNet等
        # 为简化示例，这里使用模拟数据
        
        # 模拟分类结果
        classifications = [
            {"label": "室内场景", "confidence": 0.85},
            {"label": "办公室", "confidence": 0.72},
            {"label": "家具", "confidence": 0.65}
        ]
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "classified_image.jpg")
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        
        return {
            "status": "success",
            "summary": "图像分类完成",
            "classifications": classifications,
            "output_path": output_path
        }
    
    def _enhance_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        图像增强
        
        Args:
            image: 图像数据
            
        Returns:
            增强结果
        """
        # 进行简单的图像增强
        # 调整亮度和对比度
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        # 锐化图像
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "enhanced_image.jpg")
        enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_image_bgr)
        
        return {
            "status": "success",
            "summary": "图像增强完成",
            "enhancements": ["亮度调整", "对比度调整", "锐化"],
            "output_path": output_path
        }
    
    def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        人脸检测
        
        Args:
            image: 图像数据
            
        Returns:
            检测结果
        """
        # 这里应该使用人脸检测模型，如Haar级联分类器、DNN等
        # 为简化示例，这里使用OpenCV的Haar级联分类器
        
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 加载人脸检测器
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            # 如果找不到级联分类器文件，使用模拟数据
            faces = [
                {"x": 100, "y": 150, "width": 200, "height": 200},
                {"x": 400, "y": 200, "width": 180, "height": 180}
            ]
        else:
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # 转换检测结果格式
            faces = []
            for (x, y, w, h) in faces_rect:
                faces.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        
        # 在图像上标记人脸
        result_image = image.copy()
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 保存处理后的图像
        output_path = os.path.join(self.output_dir, "detected_faces.jpg")
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_bgr)
        
        return {
            "status": "success",
            "summary": f"检测到{len(faces)}张人脸",
            "faces": faces,
            "output_path": output_path
        }
    
    def get_supported_features(self) -> List[str]:
        """
        获取支持的图像处理功能列表
        
        Returns:
            功能列表
        """
        return self.supported_features