#include "esp_camera.h"
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
#include "camera_pins.h"
#include <WiFi.h>
#include <HTTPClient.h>
#define RXD 14  // UART RX 接腳 (GPIO 16)
#define TXD 15  // UART TX 接腳 (GPIO 17)
#define UART_BAUD_RATE 9600

// Wi-Fi 資訊
const char *ssid = "***";
const char *password = "***";
const char* serverUrl = "https://------/upload"; // Colab 的伺服器 URL
const char* serverUr2 = "https://------/send_data"; // Colab 的伺服器 URL

void setup() {
  Serial.begin(115200);
  Serial1.begin(UART_BAUD_RATE, SERIAL_8N1, RXD, TXD);
  WiFi.begin(ssid, password);
  Serial.println(ssid);
  //Serial.setDebugOutput(true);
  //Serial.println();
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("正在連接 Wi-Fi...");
  }
  Serial.println("Wi-Fi 已連接！");

  // 初始化相機
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_SVGA; //800X600
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_240X240; //240X240
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_SVGA); //800X600
  }

// Setup LED FLash if LED pin is defined in camera_pins.h
// #if defined(LED_GPIO_NUM)
// setupLedFlash(LED_GPIO_NUM);
// #endif
}

void loop() {
  Serial.println("Capturing image...");

  // 拍攝照片
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.printf("Width: %d, Height: %d, Format: %d\n", fb->width, fb->height, fb->format);
  // 傳輸影像
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "image/jpeg");
    //http.addHeader("User-Agent", "ESP32-Camera/1.0");
    int httpResponseCode = http.sendRequest("POST", fb->buf, fb->len);
    Serial.println("影像大小: " + String(fb->len) + " bytes");
    if (fb->buf[0] == 0xFF && fb->buf[1] == 0xD8) {
      Serial.println("JPEG 標頭有效");
    } else {
      Serial.println("JPEG 標頭無效");
    }

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("伺服器回應: " + response);
    } else {
      Serial.println("傳輸失敗，錯誤代碼: " + String(httpResponseCode));
    }
    http.end();
  }
  esp_camera_fb_return(fb);
  // 延遲 5 秒
  delay(5000);
  
  if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUr2);
        int httpResponseCode = http.GET();        
        if (httpResponseCode > 0) {
          String response = http.getString(); // 取得伺服器回應
          String light = String(response[23]);  // 將 char 轉為 String
          String num1 = String(response[42]);
          String num2 = String(response[61]);
          String can_go = String(response[11]);
    
    // 拼接字符串
          String result = light + num1 + num2 + can_go;
          Serial.println("Get: " + response);
          Serial.println(result);
          Serial1.println(result);
        } else {
          Serial.println("傳輸失敗，錯誤代碼: " + String(httpResponseCode));
        }
        http.end();
  }
}