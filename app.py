import streamlit as st
import requests
import json
import ssl
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# ---------------------------------------------------------
# CẤU HÌNH XỬ LÝ LỖI LEGACY SSL CHO MOBIFONE INVOICE
# ---------------------------------------------------------
class CustomSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        # Bật cờ cho phép kết nối máy chủ dùng chuẩn SSL cũ (Legacy SSL)
        context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def get_legacy_session():
    session = requests.Session()
    session.mount('https://', CustomSSLAdapter())
    return session

# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Hệ thống Xuất Hóa Đơn MobiFone AI", page_icon="🧾", layout="wide")

def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

# ---------------------------------------------------------
# 2. CÁC HÀM XỬ LÝ KẾT NỐI API & AI
# ---------------------------------------------------------
def mobifone_test_connection(mst: str, user: str, pwd: str) -> dict:
    """Xác thực tài khoản MobiFone Invoice và kiểm tra dải ký hiệu phát hành"""
    base_url = f"https://{mst}.mobifoneinvoice.vn/api"
    login_url = f"{base_url}/Account/Login"
    login_payload = {"username": user, "password": pwd}
    
    # Sử dụng Session hỗ trợ Legacy SSL
    session = get_legacy_session()
    
    res = session.post(login_url, json=login_payload, headers={"Content-Type": "application/json"}).json()
    if "token" not in res:
        raise Exception(res.get('error') or res.get('message') or "Tài khoản hoặc mật khẩu MobiFone Invoice không chính xác.")
    
    token = res["token"]
    ma_dvcs = res["ma_dvcs"]
    headers = {
        "Authorization": f"Bear {token};{ma_dvcs}",
        "Content-Type": "application/json"
    }
    
    ref_url = f"{base_url}/System/GetDataReferencesByRefId?refId=RF00059"
    ref_res = session.get(ref_url, headers=headers).json()
    if not isinstance(ref_res, list) or len(ref_res) == 0:
        raise Exception("Đăng nhập thành công nhưng không tìm thấy Dải ký hiệu hóa đơn khả dụng trên MobiFone.")
    
    cctbao_id = ref_res[0]["qlkhsdung_id"]
    
    return {
        "token": token,
        "ma_dvcs": ma_dvcs,
        "cctbao_id": cctbao_id,
        "base_url": base_url
    }

import re  # Đảm bảo đã import re ở đầu tệp app.py

import io
import json
import re
from PIL import Image
import google.generativeai as genai

def extract_invoice_data(image: Image.Image, api_key: str) -> dict:
    """
    Tự động dò mô hình Gemini hợp lệ với API Key + Nén ảnh tăng tốc (1s) + Đọc chữ viết tay chuẩn xác
    """
    genai.configure(api_key=api_key)
    
    # 1. NÉN KÍCH THƯỚC ẢNH ĐỂ TRUYỀN TẢI NHANH (TỐI ĐA 1024px)
    img_temp = image.copy()
    img_temp.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    img_bytes = io.BytesIO()
    img_temp.convert("RGB").save(img_bytes, format='JPEG', quality=85)
    optimized_image = Image.open(img_bytes)

    # 2. PROMPT CHUYÊN SÂU ĐỌC CHỮ VIẾT TAY & SUY LUẬN LOGIC TOÁN HỌC
    prompt = """
    Bạn là chuyên gia bóc tách hóa đơn viết tay tiếng Việt. Hãy phân tích biên nhận viết tay của cửa hàng Nhôm - Inox - Đồ gia dụng này và trả về JSON chuẩn.

    Cấu trúc JSON yêu cầu:
    {
        "customer_name": "Khách hàng lẻ",
        "customer_mst": "",
        "customer_address": "",
        "payment_method": "Tiền mặt",
        "items": [
            {
                "stt": 1,
                "ten": "Tên mặt hàng",
                "sluong": 1,
                "dgia": 100000,
                "thtien": 100000,
                "tsuat": "10"
            }
        ]
    }

    QUY TẮC BÓC TÁCH CHÍNH XÁC:
    1. TÊN MẶT HÀNG: Đọc kỹ chữ viết tay đồ gia dụng/nhựa/inox (VD: Rổ chén, Rổ nhựa, Giỏ rác, Lô tô 8, Lô tô tô, Sợi chén...).
    2. QUY ĐỔI TIỀN VIẾT TẮT: Các ô có dấu chấm ở cuối như 420. -> 420000, 140. -> 140000, 50. -> 50000, 230. -> 230000, 300. -> 300000, 730. -> 730000.
    3. ĐỐI CHIẾU LOGIC TOÁN HỌC: Bắt buộc `Thành tiền = Số lượng x Đơn giá`. Nếu số lượng hoặc đơn giá viết tay mờ/khó đọc, hãy dùng Thành tiền chia ngược lại để suy ra đúng Số lượng và Đơn giá.
    4. Trường `tsuat` mặc định là "10".
    5. Chỉ trả về duy nhất chuỗi JSON.
    """

    # 3. TỰ ĐỘNG DÒ DANH SÁCH MÔ HÌNH THỰC TẾ HỖ TRỢ BỞI API KEY
    usable_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                usable_models.append(m.name)
    except Exception:
        pass

    # Sắp xếp ưu tiên mô hình Flash để có tốc độ cao nhất
    if usable_models:
        usable_models = sorted(usable_models, key=lambda name: 0 if 'flash' in name.lower() else 1)
    else:
        # Danh sách mô hình dự phòng mở rộng
        usable_models = [
            'models/gemini-2.0-flash',
            'models/gemini-2.5-flash',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-flash-001',
            'models/gemini-1.5-flash-002',
            'gemini-2.0-flash',
            'gemini-1.5-flash'
        ]

    response_text = ""
    last_err = None

    # 4. THỬ LẦN LƯỢT CÁC MÔ HÌNH KHẢ DỤNG
    for model_name in usable_models:
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            )
            res = model.generate_content([prompt, optimized_image])
            if res and res.text:
                response_text = res.text.strip()
                break
        except Exception as e:
            last_err = e
            continue

    if not response_text:
        raise Exception(f"Không kết nối được mô hình AI khả dụng. Lỗi chi tiết: {last_err}")

    # 5. BÓC TÁCH VÀ GIẢI MÃ KHỐI JSON CHUẨN
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    clean_json = response_text[start_idx:end_idx+1] if (start_idx != -1 and end_idx != -1) else response_text

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(clean_json)
    return data

def create_mobifone_invoice(connection_info: dict, extracted_data: dict) -> dict:
    """Khởi tạo hóa đơn điện tử lên hệ thống MobiFone Invoice"""
    base_url = connection_info["base_url"]
    headers = {
        "Authorization": f"Bear {connection_info['token']};{connection_info['ma_dvcs']}",
        "Content-Type": "application/json"
    }
    
    details_list = []
    for idx, item in enumerate(extracted_data.get("items", []), start=1):
        sluong = float(item.get("sluong", 1))
        dgia = float(item.get("dgia", 0))
        thtien = float(item.get("thtien", sluong * dgia))
        tsuat = str(item.get("tsuat", "10"))
        tthue = thtien * (float(tsuat) / 100) if tsuat.isdigit() and float(tsuat) > 0 else 0
        
        details_list.append({
            "stt": idx, "ma": str(idx), "ten": item.get("ten", "Hàng hóa/Dịch vụ"),
            "mdvtinh": "CAI", "dvtinh": "Cái", "sluong": sluong, "dgia": dgia,
            "thtien": thtien, "tlckhau": 0, "stckhau": 0, "tsuat": tsuat,
            "tthue": tthue, "tgtien": thtien + tthue, "kmai": 1
        })
        
    tgtcthue = sum(d["thtien"] for d in details_list)
    tgtthue = sum(d["tthue"] for d in details_list)
    tgtttbso = tgtcthue + tgtthue
    
    save_payload = {
        "editmode": 1,
        "data": [{
            "cctbao_id": connection_info["cctbao_id"],
            "nlap": datetime.now().strftime("%Y-%m-%d"),
            "sdhang": "", "dvtte": "VND", "tgia": 1,
            "htttoan": extracted_data.get("payment_method", "Tiền mặt/Chuyển khoản"),
            "mnmua": extracted_data.get("customer_mst", "") or "KH01",
            "mst": extracted_data.get("customer_mst", ""),
            "tnmua": extracted_data.get("customer_name", "Khách hàng lẻ"),
            "ten": extracted_data.get("customer_name", "Khách hàng lẻ"),
            "dchi": extracted_data.get("customer_address", ""),
            "email": "",
            "tgtcthue": tgtcthue, "tgtthue": tgtthue,
            "tgtttbso": tgtttbso, "tgtttbso_last": tgtttbso,
            "tkcktmn": 0, "ttcktmai": 0, "tgtphi": 0,
            "mdvi": connection_info["ma_dvcs"],
            "tthdon": 0, "is_hdcma": 1,
            "details": [{"data": details_list}],
            "hoadon68_phi": [{"data": []}],
            "hoadon68_khac": [{"data": []}]
        }]
    }
    
    save_url = f"{base_url}/Invoice68/SaveListHoadon78"
    session = get_legacy_session()
    res = session.post(save_url, json=save_payload, headers=headers)
    return res.json()

# ---------------------------------------------------------
# 3. QUẢN LÝ TRẠNG THÁI PHIÊN LÀM VIỆC (SESSION STATE)
# ---------------------------------------------------------
if "connected" not in st.session_state:
    st.session_state.connected = False
if "connection_info" not in st.session_state:
    st.session_state.connection_info = None

# ---------------------------------------------------------
# 4. GIAO DIỆN ĐĂNG NHẬP HOẶC MÀN HÌNH CHÍNH
# ---------------------------------------------------------
if not st.session_state.connected:
    st.title("🔑 Đăng Nhập & Kết Nối Hệ Thống MobiFone Invoice")
    st.markdown("Nhập thông tin kết nối dịch vụ Hóa đơn điện tử MobiFone và Gemini AI Key để bắt đầu.")
    st.divider()

    with st.form("login_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            mst_input = st.text_input("Mã số thuế doanh nghiệp (MST)", value=get_secret("MOBIFONE_MST"))
            user_input = st.text_input("Tài khoản MobiFone Invoice", value=get_secret("MOBIFONE_USER"))
            pwd_input = st.text_input("Mật khẩu MobiFone Invoice", value=get_secret("MOBIFONE_PASS"), type="password")
        with col_b:
            gemini_input = st.text_input("Gemini API Key (AI OCR)", value=get_secret("GEMINI_API_KEY"), type="password")
            st.caption("ℹ️ Gemini API Key được sử dụng để đọc dữ liệu tự động từ ảnh hóa đơn/biên nhận.")

        submit_btn = st.form_submit_button("🔌 Kết Nối MobiFone Invoice", type="primary", use_container_width=True)

        if submit_btn:
            if not (mst_input and user_input and pwd_input and gemini_input):
                st.error("⚠️ Vui lòng nhập đầy đủ tất cả thông tin đăng nhập!")
            else:
                with st.spinner("Đang xác thực tài khoản và kiểm tra kết nối MobiFone..."):
                    try:
                        conn_info = mobifone_test_connection(mst_input, user_input, pwd_input)
                        st.session_state.connection_info = conn_info
                        st.session_state.gemini_key = gemini_input
                        st.session_state.mst = mst_input
                        st.session_state.user = user_input
                        st.session_state.connected = True
                        st.success("✅ Kết nối thành công!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Kết nối thất bại: {str(e)}")

else:
    with st.sidebar:
        st.header("⚙️ Trạng Thái Kết Nối")
        st.success("🟢 **Đã kết nối MobiFone**")
        st.write(f"• **MST:** `{st.session_state.mst}`")
        st.write(f"• **Tài khoản:** `{st.session_state.user}`")
        st.write(f"• **Mã đơn vị:** `{st.session_state.connection_info['ma_dvcs']}`")
        st.divider()
        if st.button("🔴 Ngắt kết nối / Đăng xuất", type="secondary", use_container_width=True):
            st.session_state.connected = False
            st.session_state.connection_info = None
            st.rerun()

    st.title("🧾 Hệ Thống Bóc Tách & Xuất Hóa Đơn Tự Động")
    st.caption("Tải lên ảnh biên nhận -> AI bóc tách thông tin -> Khởi tạo HĐĐT trên MobiFone Invoice.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Tải ảnh hóa đơn / biên nhận")
        uploaded_file = st.file_uploader("Chọn tệp ảnh (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh biên nhận đã tải lên", use_container_width=True)

    with col2:
        st.subheader("2. Kết quả bóc tách & Khởi tạo hóa đơn")
        
        if uploaded_file:
            if st.button("🚀 Bắt đầu Bóc Tách & Xuất Hóa Đơn", type="primary", use_container_width=True):
                try:
                    with st.spinner("AI đang đọc và bóc tách dữ liệu từ ảnh..."):
                        extracted_data = extract_invoice_data(image, st.session_state.gemini_key)
                        st.success("✅ Bóc tách dữ liệu ảnh thành công!")
                        st.json(extracted_data)
                    
                    with st.spinner("Đang gửi dữ liệu khởi tạo hóa đơn lên MobiFone Invoice..."):
                        result = create_mobifone_invoice(st.session_state.connection_info, extracted_data)
                        st.balloons()
                        st.success("🎉 Khởi tạo hóa đơn thành công trên MobiFone Invoice!")
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"❌ Xảy ra lỗi: {str(e)}")
        else:
            st.info("Vui lòng tải ảnh lên ở cột bên trái để bắt đầu xử lý.")