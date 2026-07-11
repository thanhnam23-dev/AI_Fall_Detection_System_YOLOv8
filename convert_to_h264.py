import subprocess
import sys
import os
import argparse

def convert_video(input_path, output_path=None):
    """
    Chuyển đổi codec của một video bất kỳ sang H.264 sử dụng bộ mã hóa libopenh264 của ffmpeg.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found '{input_path}'")
        return False
        
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_h264{ext}"
        
    print(f"Converting: '{input_path}' -> '{output_path}'...")
    
    # Sử dụng libx264 vì đây là encoder tiêu chuẩn và tương thích tốt nhất trên hệ thống của bạn
    cmd = [
        'ffmpeg', '-y', 
        '-i', input_path, 
        '-vcodec', 'libx264', 
        '-f', 'mp4', 
        output_path
    ]
    
    try:
        # Chạy lệnh ffmpeg và theo dõi kết quả
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Conversion successful! Output saved at: '{output_path}'")
            return True
        else:
            print("Error occurred while running ffmpeg:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found on the system. Please install ffmpeg.")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chuyển đổi codec video sang chuẩn H.264 để có thể xem trực tiếp trên IDE/Trình duyệt.")
    parser.add_argument("input", help="Đường dẫn đến file video mp4 cần chuyển đổi")
    parser.add_argument("-o", "--output", help="Đường dẫn file kết quả (mặc định sẽ tự động thêm hậu tố _h264)", default=None)
    
    # Nếu không có tham số nào truyền vào, in hướng dẫn
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    convert_video(args.input, args.output)
