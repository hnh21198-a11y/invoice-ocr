import os
import io
import json
import re
import ssl
import requests
import urllib3
import pandas as pd
import streamlit as st
from PIL import Image
import google.generativeai as genai
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from datetime import datetime

# Tệp lưu trữ từ vựng AI đã học
DICT_FILE = "learned_dictionary.json"

# Bỏ qua cảnh báo SSL đối với máy chủ MobiFone
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CustomSSLAdapter(HTTPAdapter):
    """SSL Adapter tùy chỉnh hỗ trợ các máy chủ đời cũ và bypass kiểm tra chứng chỉ an toàn"""
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def get_mobifone_session():
    """Tạo Session HTTP để kết nối MobiFone API"""
    session = requests.Session()
    session.mount('https://', CustomSSLAdapter())
    return session

def load_learned_dictionary() -> list:
    """Đọc bộ từ vựng đã học từ tệp json cục bộ"""
    if os.path.exists(DICT_FILE):
        try:
            with open(DICT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return ["Gô Chén", "Xô Ánh", "Ghế Lỗ Ruột", "Áo Tô 8", "Áo Tô Trắng Gốc", "Sọt Chén"]

def save_learned_dictionary(new_item_names: list) -> tuple[list, int]:
    """Cập nhật và học thêm các từ vựng mới được người dùng sửa thành công"""
    current_dict = load_learned_dictionary()
    added_count = 0
    for name in new_item_names:
        clean_name = str(name).strip()
        if clean_name and len(clean_name) > 1 and clean_name not in current_dict:
            current_dict.append(clean_name)
            added_count += 1
    
    if added_count > 0:
        with open(DICT_FILE, "w", encoding="utf-8") as f:
            json.dump(current_dict, f, ensure_ascii=False, indent=2)
            
    return current_dict, added_count

def get_live_gemini_models(api_key: str) -> list:
    """Tự động lấy danh sách mô hình Gemini đang hoạt động thực tế trên API Key"""
    genai.configure(api_key=api_key)
    live_models = []
    try:
        models = genai.list_models()
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                live_models.append(m.name)
    except Exception:
        pass

    if live_models:
        live_models.sort(key=lambda name: (0 if 'flash' in name.lower() else 1, name))
    else:
        live_models = [
            "models/gemini-1.5-flash",
            "models/gemini-2.0-flash",
            "gemini-1.5-flash"
        ]
    return live_models

def extract_invoice_data(image: Image.Image, api_key: str) -> tuple[dict, str]:
    """
    Bóc tách chữ viết tay tích hợp Từ vựng Học thêm (Dynamic Few-Shot In-Context Learning)
    """
    genai.configure(api_key=api_key)
    
    img_temp = image.copy()
    img_temp.thumbnail((800, 800), Image.Resampling.LANCZOS)
    img_bytes = io.BytesIO()
    img_temp.convert("RGB").save(img_bytes, format='JPEG', quality=75)
    img_bytes.seek(0)
    optimized_image = Image.open(img_bytes)

    learned_words = load_learned_dictionary()
    learned_context = ""
    if learned_words:
        vocab_list_str = "\n".join([f"    - {w}" for w in learned_words[:40]])
        learned_context = f"\n    DANH SÁCH TÊN MẶT HÀNG THƯỜNG GẶP CỦA CỬA HÀNG (ĐÃ ĐƯỢC XÁC NHẬN CHÍNH XÁC TỪ CÁC LẦN XUẤT HÓA ĐƠN TRƯỚC):\n{vocab_list_str}\n    -> Ưu tiên đối chiếu nét chữ viết tay mờ với danh sách từ vựng trên nếu thấy khớp nghĩa."

    prompt = f"""
    Bạn là chuyên gia bóc tách hóa đơn viết tay tiếng Việt. Hãy phân tích biên nhận viết tay này và trả về duy nhất chuỗi JSON chuẩn theo cấu trúc:
    {{
        "customer_name": "Khách hàng lẻ",
        "customer_mst": "",
        "customer_address": "",
        "payment_method": "Tiền mặt/Chuyển khoản",
        "items": [
            {{
                "stt": 1,
                "ten": "Tên mặt hàng",
                "sluong": 1,
                "dgia": 100000,
                "thtien": 100000,
                "tsuat": "10"
            }}
        ]
    }}

    QUY TẮC BÓC TÁCH CHÍNH XÁC:
    1. TÊN MẶT HÀNG: Đọc kỹ chữ viết tay tiếng Việt (đồ gia dụng, nhôm inox, nhựa, nội thất...).{learned_context}
    2. QUY ĐỔI TIỀN VIẾT TẮT: Các ô số tiền có dấu chấm ở cuối như 420. -> 420000, 140. -> 140000, 50. -> 50000, 230. -> 230000, 300. -> 300000, 730. -> 730000.
    3. ĐỐI CHIẾU LOGIC TOÁN HỌC: Bắt buộc `Thành tiền = Số lượng x Đơn giá`. Nếu số mờ, lấy Thành tiền chia ngược lại để suy ra đúng Số lượng và Đơn giá.
    4. Trường `tsuat` mặc định điền "10".
    5. CHỈ TRẢ VỀ DUY NHẤT CHUỖI JSON.
    """

    candidate_models = get_live_gemini_models(api_key)
    cached_model = st.session_state.get("working_model")
    if cached_model and cached_model in candidate_models:
        candidate_models.remove(cached_model)
        candidate_models.insert(0, cached_model)

    response_text = ""
    used_model = ""
    last_err = None

    for model_name in candidate_models:
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
                used_model = model_name
                break
        except Exception as e:
            last_err = e
            continue

    if not response_text:
        raise Exception(f"Không thể kết nối Gemini AI. Lỗi chi tiết: {last_err}")

    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    clean_json = response_text[start_idx:end_idx+1] if (start_idx != -1 and end_idx != -1) else response_text

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(clean_json)
    return data, used_model

def get_cctbao_id(session: requests.Session, host: str, headers: dict) -> str:
    """Lấy ID dải ký hiệu mẫu hóa đơn (qlkhsdung_id / cctbao_id) từ MobiFone API (API 2 trong tài liệu)"""
    url = f"{host}/api/System/GetDataReferencesByRefId?refId=RF00059"
    try:
        res = session.get(url, headers=headers, verify=False, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("qlkhsdung_id") or data[0].get("id") or ""
    except Exception:
        pass
    return ""

def send_to_mobifone_api(data: dict, mst: str, username: str, password: str, madonvi: str, custom_domain: str = "") -> dict:
    """
    Kết nối API MobiFone Invoice theo đúng tài liệu kỹ thuật Ver 4.7 (Nghị định 70/123, Thông tư 78)
    """
    session = get_mobifone_session()
    
    # 1. Xác định Host (Xử lý tách đuôi chi nhánh -999 để tránh lỗi DNS getaddrinfo failed)
    hosts = []
    if custom_domain and custom_domain.strip():
        d = custom_domain.strip().rstrip('/')
        if not d.startswith("http"):
            d = f"https://{d}"
        hosts.append(d)

    clean_mst = mst.split('-')[0].strip() if mst else ""
    default_host = f"https://{clean_mst}.mobifoneinvoice.vn"
    if default_host not in hosts:
        hosts.append(default_host)

    # 2. Đăng nhập lấy Token (Chuẩn Tài liệu Ver 4.7 Trang 5: username, password)
    login_routes = [
        "/api/Account/Login",
        "/Account/Login",
        "/api/account/login"
    ]

    login_payload = {
        "username": username,
        "password": password
    }

    token = ""
    res_ma_dvcs = madonvi
    working_host = ""
    login_logs = []

    for host in hosts:
        for route in login_routes:
            url = f"{host}{route}"
            try:
                res = session.post(url, json=login_payload, verify=False, timeout=8)
                login_logs.append(f"• POST {url} -> HTTP {res.status_code}")
                if res.status_code == 200:
                    try:
                        res_json = res.json()
                        token = res_json.get("token") or res_json.get("data", {}).get("token") or ""
                        if token:
                            working_host = host
                            if res_json.get("ma_dvcs"):
                                res_ma_dvcs = res_json.get("ma_dvcs")
                            break
                    except Exception:
                        pass
            except Exception as e:
                login_logs.append(f"• POST {url} -> Lỗi: {e}")
        if token:
            break

    if not token:
        log_str = "\n".join(login_logs)
        raise Exception(f"Đăng nhập thất bại (Không lấy được Token).\nNhật ký thử nghiệm:\n{log_str}")

    # 3. Header Authorization theo chuẩn MobiFone (Tài liệu Trang 6: Bear token;ma_dvcs)
    ma_dv = res_ma_dvcs if res_ma_dvcs else madonvi
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bear {token};{ma_dv}"
    }

    # 4. Lấy dải ký hiệu hóa đơn (cctbao_id)
    cctbao_id = get_cctbao_id(session, working_host, headers)

    # 5. Đóng gói Payload theo cấu trúc MobiFone Invoice 78 chuẩn Tài liệu Ver 4.7 Trang 8-15
    details_data = []
    tgtcthue = 0.0
    tgtthue = 0.0

    for idx, item in enumerate(data.get("items", []), start=1):
        sluong = float(item.get("sluong", 1))
        dgia = float(item.get("dgia", 0))
        thtien = float(item.get("thtien", sluong * dgia))
        tsuat_str = str(item.get("tsuat", "10")).replace("%", "")
        
        try:
            tsuat_val = float(tsuat_str) if tsuat_str not in ["KCT", "KKDT", "0", "-1", "-2"] else 0.0
        except ValueError:
            tsuat_val = 10.0 if tsuat_str == "10" else 0.0
            
        tthue = thtien * (tsuat_val / 100.0)
        tgtien = thtien + tthue
        
        tgtcthue += thtien
        tgtthue += tthue

        details_data.append({
            "stt": idx,
            "ma": str(item.get("ma", idx)),
            "ten": str(item.get("ten", "")),
            "mdvtinh": "CAI",
            "dvtinh": "Cái",
            "sluong": sluong,
            "dgia": dgia,
            "thtien": thtien,
            "tlckhau": 0,
            "stckhau": 0,
            "tsuat": tsuat_str,
            "tthue": tthue,
            "tgtien": tgtien,
            "kmai": 1
        })

    tgtttbso = tgtcthue + tgtthue

    invoice_payload = {
        "editmode": 1,
        "data": [{
            "cctbao_id": cctbao_id,
            "nlap": datetime.now().strftime("%Y-%m-%d"),
            "dvtte": "VND",
            "tgia": 1,
            "htttoan": data.get("payment_method", "Tiền mặt/Chuyển khoản"),
            "mst": data.get("customer_mst", ""),
            "tnmua": data.get("customer_name", "Khách hàng lẻ"),
            "ten": data.get("customer_name", "Khách hàng lẻ"),
            "dchi": data.get("customer_address", "Hà Nội"),
            "tgtcthue": tgtcthue,
            "tgtthue": tgtthue,
            "tgtttbso": tgtttbso,
            "tgtttbso_last": tgtttbso,
            "mdvi": ma_dv,
            "tthdon": 0,
            "is_hdcma": 1,
            "details": [{
                "data": details_data
            }]
        }]
    }

    # 6. Gửi request đến Endpoint chuẩn
    save_routes = [
        "/api/Invoice68/SaveListHoadon78",     # HĐĐT GTGT TT78
        "/api/Invoice68/SaveListHoadon78MTT",  # Hóa đơn máy tính tiền
        "/api/Invoice68/SaveAndSignHoadon78"   # Khởi tạo và ký gửi
    ]

    auth_headers_to_try = [
        headers,
        {"Content-Type": "application/json", "Authorization": f"Bearer {token};{ma_dv}"},
        {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    ]

    save_logs = []
    for route in save_routes:
        save_url = f"{working_host}{route}"
        for h in auth_headers_to_try:
            try:
                save_res = session.post(save_url, json=invoice_payload, headers=h, verify=False, timeout=12)
                save_logs.append(f"• POST {save_url} -> HTTP {save_res.status_code}")
                
                if save_res.status_code == 200:
                    try:
                        return save_res.json()
                    except Exception:
                        return {"status": "success", "message": "Đã gửi lệnh tạo hóa đơn thành công!", "response": save_res.text}
            except Exception as e:
                save_logs.append(f"• POST {save_url} -> Lỗi: {e}")

    save_log_str = "\n".join(save_logs)
    raise Exception(f"Không thể khởi tạo hóa đơn tại máy chủ MobiFone ({working_host}).\nNhật ký thử nghiệm:\n{save_log_str}")

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Hệ Thống Bóc Tách & Xuất Hóa Đơn Tự Động", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "working_model" not in st.session_state:
    st.session_state.working_model = "Tự động kết nối..."
if "custom_domain" not in st.session_state:
    st.session_state.custom_domain = ""

# 1. MÀN HÌNH ĐĂNG NHẬP
if not st.session_state.logged_in:
    st.title("🔒 Đăng Nhập Hệ Thống (MobiFone Invoice & AI Engine)")
    with st.form("login_form"):
        mst = st.text_input("Mã số thuế doanh nghiệp (MST)", value="0100686209-999")
        username = st.text_input("Tài khoản MobiFone Invoice", value="ADMINISTRATOR")
        password = st.text_input("Mật khẩu MobiFone", type="password")
        madonvi = st.text_input("Mã đơn vị / Mã cơ sở (Nội bộ)", value="VP")
        custom_domain = st.text_input("Server Domain MobiFone (Tùy chọn)", placeholder="Ví dụ: 0100686209.mobifoneinvoice.vn")
        gemini_key = st.text_input("Google Gemini API Key", type="password")
        submit = st.form_submit_button("Đăng nhập", type="primary")

        if submit:
            if mst and username and password and gemini_key:
                st.session_state.logged_in = True
                st.session_state.mst = mst
                st.session_state.username = username
                st.session_state.password = password
                st.session_state.madonvi = madonvi
                st.session_state.custom_domain = custom_domain
                st.session_state.gemini_key = gemini_key
                st.session_state.working_model = "Tự động kết nối..."
                st.success("Đăng nhập hệ thống thành công!")
                st.rerun()
            else:
                st.error("Vui lòng điền đầy đủ thông tin đăng nhập!")

# 2. MÀN HÌNH CHÍNH
else:
    st.sidebar.title("⚙️ Trạng Thái Kết Nối")
    st.sidebar.success("Đã kết nối MobiFone")
    st.sidebar.write(f"• **MST:** {st.session_state.get('mst', '')}")
    st.sidebar.write(f"• **Tài khoản:** {st.session_state.get('username', '')}")
    
    clean_domain = st.session_state.get('mst', '').split('-')[0].strip()
    server_display = st.session_state.get("custom_domain", "") or f"{clean_domain}.mobifoneinvoice.vn"
    st.sidebar.write(f"• **Server:** `{server_display}`")
    st.sidebar.write(f"• **Mô hình AI Active:** `{st.session_state.get('working_model', 'N/A')}`")

    learned_words = load_learned_dictionary()
    st.sidebar.divider()
    st.sidebar.subheader(f"🧠 Bộ Từ Vựng AI ({len(learned_words)} từ)")
    with st.sidebar.expander("Xem danh sách từ đã học"):
        for word in learned_words:
            st.write(f"- {word}")
        if st.button("🗑️ Xóa bộ nhớ từ vựng"):
            if os.path.exists(DICT_FILE):
                os.remove(DICT_FILE)
            st.rerun()

    if st.sidebar.button("🔴 Ngắt kết nối / Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.extracted_data = None
        st.session_state.working_model = "Tự động kết nối..."
        st.session_state.custom_domain = ""
        st.rerun()

    st.title("🧾 Hệ Thống Bóc Tách & Xuất Hóa Đơn Tự Động")
    st.caption("Tải lên ảnh -> AI bóc tách (Tự học từ vựng) -> Xem & Sửa dữ liệu -> Khởi tạo HĐĐT MobiFone Invoice.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. Tải ảnh hóa đơn / biên nhận")
        uploaded_file = st.file_uploader("Chọn tệp ảnh (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh hóa đơn đã tải lên", use_container_width=True)
            
            if st.button("🔍 Bóc Tách Dữ Liệu (AI)", type="primary", use_container_width=True):
                with st.spinner("AI đang bóc tách chữ viết tay và đối chiếu từ vựng đã học..."):
                    try:
                        extracted, used_model = extract_invoice_data(
                            image, 
                            st.session_state.gemini_key
                        )
                        st.session_state.extracted_data = extracted
                        st.session_state.working_model = used_model
                        st.success(f"✅ Bóc tách thành công qua mô hình [{used_model}]!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Xảy ra lỗi bóc tách: {e}")

    with col2:
        st.subheader("2. Xem trước & Chỉnh sửa dữ liệu")
        
        if st.session_state.extracted_data is None:
            st.info("👈 Vui lòng tải ảnh lên và nhấn **🔍 Bóc Tách Dữ Liệu (AI)** để xem trước thông tin hóa đơn tại đây.")
        else:
            data = st.session_state.extracted_data
            
            with st.form("invoice_editor_form"):
                st.markdown("##### 📌 Thông tin khách hàng")
                c1, c2 = st.columns(2)
                with c1:
                    cust_name = st.text_input("Tên người mua / Khách hàng", value=data.get("customer_name", "Khách hàng lẻ"))
                    cust_mst = st.text_input("Mã số thuế người mua", value=data.get("customer_mst", ""))
                with c2:
                    cust_address = st.text_input("Địa chỉ người mua", value=data.get("customer_address", ""))
                    pay_method = st.selectbox("Hình thức thanh toán", ["Tiền mặt/Chuyển khoản", "Tiền mặt", "Chuyển khoản"], index=0)

                st.markdown("##### 🛒 Danh sách hàng hóa / Dịch vụ (Chỉnh sửa trực tiếp)")
                
                items_list = data.get("items", [])
                items_df = pd.DataFrame(items_list)
                
                expected_cols = ["stt", "ten", "sluong", "dgia", "thtien", "tsuat"]
                for col in expected_cols:
                    if col not in items_df.columns:
                        items_df[col] = ""

                edited_df = st.data_editor(
                    items_df[expected_cols],
                    num_rows="dynamic",
                    column_config={
                        "stt": st.column_config.NumberColumn("STT", width="small"),
                        "ten": st.column_config.TextColumn("Tên mặt hàng", width="large", required=True),
                        "sluong": st.column_config.NumberColumn("Số lượng", min_value=1, default=1),
                        "dgia": st.column_config.NumberColumn("Đơn giá", format="%d", default=0),
                        "thtien": st.column_config.NumberColumn("Thành tiền", format="%d", default=0),
                        "tsuat": st.column_config.SelectboxColumn("Thuế suất (%)", options=["0", "5", "8", "10", "KCT", "KKDT"], default="10")
                    },
                    use_container_width=True
                )

                st.divider()
                
                btn_issue = st.form_submit_button("🚀 XÁC NHẬN & XUẤT HÓA ĐƠN MOBIFONE", type="primary", use_container_width=True)

                if btn_issue:
                    final_items = edited_df.to_dict("records")
                    
                    new_item_names = [item.get("ten", "") for item in final_items if item.get("ten")]
                    updated_dict, added_count = save_learned_dictionary(new_item_names)
                    
                    if added_count > 0:
                        st.toast(f"🧠 AI đã học thêm {added_count} từ vựng mặt hàng mới vào bộ nhớ!", icon="💡")

                    final_data = {
                        "customer_name": cust_name,
                        "customer_mst": cust_mst,
                        "customer_address": cust_address,
                        "payment_method": pay_method,
                        "items": final_items
                    }

                    with st.spinner("Đang kết nối & khởi tạo hóa đơn trên MobiFone Invoice..."):
                        try:
                            mobifone_res = send_to_mobifone_api(
                                final_data,
                                st.session_state.mst,
                                st.session_state.username,
                                st.session_state.password,
                                st.session_state.madonvi,
                                st.session_state.get("custom_domain", "")
                            )
                            st.balloons()
                            st.success("🎉 Khởi tạo Hóa đơn điện tử trên MobiFone Invoice THÀNH CÔNG!")
                            st.write("**Kết quả phản hồi từ máy chủ MobiFone:**")
                            st.json(mobifone_res)
                        except Exception as e:
                            st.error(f"❌ Khởi tạo hóa đơn thất bại: {e}")
