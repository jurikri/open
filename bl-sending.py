import serial # pip install pyserial
import time

# 시리얼 포트 설정
port = "COM11"  # 포트 번호
baudrate = 9600  # 보드레이트

def ms_blsend(data):
    ser = serial.Serial(port, baudrate)
    try:
        ser.write(data.encode('ascii'))
        print("Data sent:", data)

        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting).decode('ascii')
            print("Received:", response)
    
    except Exception as e:
        print("Error:", e)
    
    finally:
        # 연결 종료
        ser.close()

if True: # 5초간 VNS 자극 후, 종료
    ms_blsend("CS112E")
    time.sleep(5)
    ms_blsend("CT110E")
# ms_blsend("CS227E")
