# 安裝套件
!pip install pyngrok
import tensorflow as tf
import numpy as np
import cv2
import math
from PIL import Image
from pyngrok import ngrok
from flask import Flask, jsonify, request, session
import json

# 解壓縮
# 先自行上傳 ssd-mobilenet_v2 至 雲端資料夾
!tar -xzvf '/content/models/ssd_mobilenet_v2_640x640.gz' -C '/content/models'

# 物件擷取
def load_model(path):

    # 使用tf.saved_model.load下載模型
    model = tf.saved_model.load(path)
    f = model.signatures['serving_default']

    #input format (1, None, None, 3)
    #print('Input_format:',f.structured_input_signature)
    #output format
    #print('Output_format:',f.structured_outputs)

    return f

def detect_multi_object(image_path, score_threshold):

    image_np = cv2.imread(image_path)
    c_image_np = image_np.astype(np.uint8)

    # 延展維度至[1, x, x, 3]
    image_np_expanded = np.expand_dims(c_image_np, axis=0)
    output_dict = detection_graph(input_tensor = image_np_expanded)

    boxes = output_dict['detection_boxes'][0].numpy() # dim = (100,4)
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy().astype(np.int64)
    num_detections = output_dict['num_detections'][0].numpy()

    # traffic light's label
    # target_class = 10
    sel_id = np.logical_and(classes == 10, scores > score_threshold)

    return boxes[sel_id]

def crop_roi_image(image_path, sel_box):
    
    image_np = cv2.imread(image_path)
    im_height, im_width, _ = image_np.shape
    (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)

    cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image, left, right

# 顏色辨識
green_lower = np.array([80, 120, 150])
green_upper = np.array([100, 170, 255])

red_lower = np.array([160, 180, 150])
red_upper = np.array([190, 255, 255])

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def load_model_mnist(model_path):
    
    # 載入 .keras 模型
    return tf.keras.models.load_model(model_path)

def preprocess_image(cropped_image):
    
    # 只保留圖片的上半部份
    height, width, _ = cropped_image.shape  # 提取高度和寬度
    upper_half_image = cropped_image[:height // 2, :]  # 取上半部

    # 計算保留區域 (中間的 80%)
    crop_margin_height = int(upper_half_image.shape[0] * 0.10)  # 上下各裁剪 10%
    crop_margin_width = int(upper_half_image.shape[1] * 0.10)  # 左右各裁剪 10%

    upper_half_image = upper_half_image[
        crop_margin_height:-crop_margin_height,  # 裁剪上下
        crop_margin_width:-crop_margin_width   # 裁剪左右
    ]
    return upper_half_image

def binarize_image(image):
    
    # 將 BGR 圖片轉換為 HSV 色彩空間
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2HSV)

    # 設定橘黃色範圍（Hue, Saturation, Value）
    lower_bound = np.array([10, 100, 100])  # 根據需求調整
    upper_bound = np.array([30, 255, 255])  # 根據需求調整

    # 建立遮罩
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # 生成二值化結果
    binary_image = mask

    return binary_image

def calculate_vertical_projection(binary_image):
    
    # 計算垂直投影
    projection = np.sum(binary_image == 255, axis=0)
    # 使用高斯平滑處理投影曲線，減少雜訊影響
    projection_smooth = cv2.GaussianBlur(projection.astype(np.float32), (5,1), 0)
    return projection_smooth

def segment_digits(projection, left, right):
   
   segments = []
   width = right - left
   min_segment_width = int(width * 0.15)  # 最小段寬
   max_segment_width = int(width * 0.7)   # 最大段寬

   # 降低閾值以更容易檢測到數字
   threshold = np.mean(projection) * 0.15

   in_segment = False
   start = None

   # 第一次掃描找出所有可能的段
   for x, value in enumerate(projection):
       if value > threshold and not in_segment:
           in_segment = True
           start = x
       elif (value <= threshold or x == len(projection)-1) and in_segment:
           end = x if value <= threshold else x+1
           if min_segment_width <= end - start <= max_segment_width:
               segments.append((max(0, start-2), min(len(projection), end+2)))
           in_segment = False

   # 如果沒有找到段，調低閾值重試
   if not segments:
       threshold = np.mean(projection) * 0.1
       for x, value in enumerate(projection):
           if value > threshold and not in_segment:
               in_segment = True
               start = x
           elif (value <= threshold or x == len(projection)-1) and in_segment:
               end = x if value <= threshold else x+1
               if min_segment_width <= end - start <= max_segment_width:
                   segments.append((max(0, start-2), min(len(projection), end+2)))
               in_segment = False

   # 合併過近的段或分割過寬的段
   if len(segments) > 2:
       merged = []
       for i in range(0, len(segments)-1, 2):
           curr_start, curr_end = segments[i]
           if i+1 < len(segments):
               next_start, next_end = segments[i+1]
               if next_start - curr_end < min_segment_width:
                   merged.append((curr_start, next_end))
               else:
                   merged.append((curr_start, curr_end))
                   merged.append((next_start, next_end))
           else:
               merged.append((curr_start, curr_end))
       segments = merged[:2]

   return segments

def resize_image_segment(binary_image, start, end, dst_width, dst_height):
    
    # 根據區段的左右邊界提取數字區段
    digit_segment = binary_image[:, start:end]  # 提取數字區段

    # 調整大小
    resized_image = cv2.resize(digit_segment, (dst_width, dst_height), interpolation=cv2.INTER_AREA)

    return resized_image

def calculate_distance(left , right , actual_width=30.0):

    # 固定的圖像寬度和高度 (拍攝時所用的解析度)
    img_width = 800  # 寬度

    # 確保邊界框的座標在圖像範圍內
    x_min = max(0, left)
    x_max = min(img_width, right)

    # 行人號誌的像素寬度
    pixel_width = right - left

    if pixel_width <= 0:
        raise ValueError("Invalid pedestrian box dimensions, width must be greater than 0.")

    # OV2640 的水平視角 (FOV) 為 68 度，計算焦距
    fov_horizontal = 68.0  # 單位：度
    focal_length = img_width / (2.0 * math.tan(math.radians(fov_horizontal) / 2.0))  # 單位：像素

    # 計算距離 (公式)
    distance = (actual_width * focal_length) / pixel_width

    return distance

# ssd_mobilenet_v2_640x640 model
saved_path = "/content/models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model"
detection_graph = load_model(saved_path)

# mobilenet mnist model
MODEL_PATH = "/content/models/mobilenet_mnist.keras"
model = load_model_mnist(MODEL_PATH)
return_data = np.array([])

# 自行建立並輸入ngrok_token
port = 5000
ngrok.set_auth_token(ngrok_token)
app = Flask(__name__)

return_data = np.array([])
return_data2 = np.array([])

@app.route("/upload", methods=['POST'])
def root():
  # 從請求中提取數據
  try:
    global return_data
    global return_data2
    image_data = request.data  # 如果直接處理二進制數據
    with open("received_image.jpg", "wb") as f:
      f.write(image_data)  # 保存為 JPEG 文件

      # 開始影像處理
      # ------------物件偵測開始------------
      image_path = "received_image.jpg"
      return_data = np.array([])
      boxes = detect_multi_object(image_path, 0.2)
      if len(boxes)>0: # 偵測到traffic light
        cropped_image , left_n , right_n = crop_roi_image(image_path,boxes[0])
        cv2.imwrite('output_image.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)) #------#
        # ------------物件偵測結束------------

        # ------------顏色辨識開始------------
        # 使用 inRange 來篩選顏色
        print("偵測物件通過")
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        red_mask = cv2.inRange(hsv_image, red_lower, red_upper)

        # 計算面積
        green_area = cv2.countNonZero(green_mask)
        red_area = cv2.countNonZero(red_mask)

        # 判斷紅綠燈顏色
        if green_area > red_area:
          print("green light")
          light = "2"
        else:
          print("red light")
          light = "1"
        # ------------顏色辨識結束------------

        # ------------偵測數字開始------------
        # 載入並預處理圖片
        print("顏色辨識通過")
        input_image = preprocess_image(cropped_image) #------#
        binary_image = binarize_image(input_image)

        # 計算垂直投影
        projection = calculate_vertical_projection(binary_image)

        # 分割數字區塊
        segments = segment_digits(projection , left_n , right_n)
        print("切割完成通過")
        if len(segments) == 0:
          print("未偵測到任何數字區塊！")
          return_data = np.array(["0","0","0"])

        for start, end in segments:
        # 提取每個數字區段
          digit_resized = resize_image_segment(binary_image, start, end, IMAGE_WIDTH, IMAGE_HEIGHT)

          # 調整至模型輸入格式
          digit_resized = digit_resized.astype(np.float32) / 255.0  # 正規化
          digit_resized = np.expand_dims(digit_resized, axis=(0, -1))  # 添加 batch 和 channel 維度

          # 推論
          predictions = model.predict(digit_resized)

          # 取得預測結果
          predicted_digit = np.argmax(predictions)
          return_data = np.append(return_data , str(int(predicted_digit)))

          print("辨識結果:",int(predicted_digit))

        if (len(return_data) == 1) :
          return_data = np.append("0" , return_data)
      # ------------偵測數字結束------------

      # ------------偵測號誌距離開始------------
        if (return_data[0]!="0" or return_data[1]!="0"):
          try:
            distance = calculate_distance(left_n , right_n , actual_width=30.0) / 100
            print(f"Estimated distance to pedestrian signal: {distance:.2f} m")
            time = int(return_data[0]) * 10 + int(return_data[1])
            if (distance / time < 0.8):
              print("通過最低速率(m/s)" , distance / time)
              return_data = np.append(return_data , "1")
            else :
              return_data = np.append(return_data  , "0")
          except ValueError as e:
            print(f"Error: {e}")

        return_data = np.append(light, return_data)

      else: # 偵測物件
        print("無traffic light物件")
        return_data = np.array(["0","0","0","0"])
      return_data2 = return_data
      return "Image received successfully", 200

  except Exception as e:
    return f"Error: {str(e)}", 500

@app.route('/send_data', methods=['GET'])
def send_data():
  global return_data2
  # 要回傳的資料
  if len(return_data2) == 0:
    return_data2 = np.array(["0","0","0","0"])
  data = {'light' : return_data2[0], 'predict_num1': return_data2[1] ,
          'predict_num2': return_data2[2] , 'can_go' : return_data2[3]}
  # 使用 jsonify() 回傳 JSON 格式資料
  return jsonify(data)
if __name__ == "__main__":
  try:
    public_url = ngrok.connect(port).public_url
    print(public_url)
    app.run(port=port)
  finally:

    ngrok.disconnect(public_url=public_url)
