import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    """Bộ xử lý yêu cầu file tĩnh, trỏ trực tiếp đến thư mục web."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    # Kiểm tra thư mục web tồn tại
    if not os.path.exists(DIRECTORY):
        print(f"Lỗi: Không tìm thấy thư mục '{DIRECTORY}' chứa tệp tin giao diện tĩnh.")
        print("Vui lòng đảm bảo cấu trúc thư mục 'web/' bao gồm index.html, css/ và js/ đã được tạo đầy đủ.")
        sys.exit(1)
        
    print("==================================================")
    print("    KHỞI ĐỘNG FRONTEND WEB SERVER - SENTINEL ALPHA")
    print("==================================================")
    print(f" Đang phục vụ các tệp tin trong thư mục: {DIRECTORY}/")
    print(f" Địa chỉ truy cập Web: http://localhost:{PORT}")
    print(" Nhấn CTRL+C để tắt Web Server.")
    print("==================================================")

    # Khởi chạy bộ lắng nghe TCP
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Đang tắt] Đã nhận tín hiệu tắt Web Frontend Server. Tạm biệt!")
    except Exception as e:
        print(f"\n[Lỗi] Không thể khởi động máy chủ: {str(e)}")

if __name__ == "__main__":
    start_server()
