# app_streamlit_classroom.py
import streamlit as st
import pandas as pd
import tempfile
import ast
import nbformat
import docx
import requests
import os
import json
import base64
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
import base64




# Optional: LLM clients (Gemini/OpenAI). We'll attempt to import but not required at dev time.
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# Google Classroom libs
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pickle
from urllib.parse import urlparse

load_dotenv()

# ---------- Config ----------
st.set_page_config(page_title="AI Auto-Grader (Classroom)", page_icon="🤖", layout="wide")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Helpers: Requirements / Code loading ----------

def send_email(service, sender, to, subject, message_text):
    """Send an email using the Gmail API"""
    try:
        message = MIMEText(message_text, "plain", "utf-8")
        message["to"] = to
        message["from"] = sender
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        body = {"raw": raw}
        sent = service.users().messages().send(userId="me", body=body).execute()
        return sent
    except HttpError as e:
        print("Error sending email:", e)
        return None
    
def extract_requirements_from_docx(path):
    doc = docx.Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def code_from_py_or_ipynb_fileobj(uploaded_file):
    name = uploaded_file.name
    content = uploaded_file.read()
    if name.endswith(".py"):
        return content.decode("utf-8", errors="ignore")
    elif name.endswith(".ipynb"):
        nb = nbformat.reads(content.decode("utf-8", errors="ignore"), as_version=4)
        return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    else:
        raise ValueError("Unsupported file type")

def code_from_path(path):
    if path.endswith(".py"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".ipynb"):
        nb = nbformat.read(path, as_version=4)
        return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    else:
        raise ValueError("Unsupported file type")

# ---------- Static analysis (AST) ----------
def analyze_code_statics(code_text):
    """
    Return dict: defined functions, called (filtered), missing (non-builtins).
    """
    try:
        tree = ast.parse(code_text)
    except Exception:
        return {"defined": [], "called": [], "missing": []}

    defined = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    called = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            # handle names and attribute calls (self.func)
            if isinstance(n.func, ast.Name):
                called.append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                called.append(n.func.attr)

    # ignore common builtins / libs that we don't treat as missing functions
    ignore = {
        "print","input","open","list","any","float","str","int","len",
        "set_option","DataFrame","tabulate","DictReader","DictWriter",
        "writerow","writeheader","exists","remove","lower","strip"
    }
    called_filtered = [c for c in called if c not in ignore]
    missing = [c for c in called_filtered if c not in defined]
    return {"defined": sorted(set(defined)), "called": sorted(set(called_filtered)), "missing": sorted(set(missing))}

# ---------- GitHub helpers ----------
def repo_api_contents_url(repo_url):
    # accept https://github.com/user/repo or with .git; return api contents endpoint
    repo_url = repo_url.rstrip(".git").rstrip("/")
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if not path:
        return None
    owner_repo = path.split("/")[:2]
    if len(owner_repo) < 2:
        return None
    owner, repo = owner_repo[0], owner_repo[1]
    return f"https://api.github.com/repos/{owner}/{repo}/contents/"



def fetch_first_code_file_from_repo(repo_url, github_token=None):
    """
    Fetch the first .py or .ipynb file in a GitHub repo (recursive search).
    Supports normal GitHub URLs (no need to manually edit links).
    """
    # تنظيف اللينك
    repo_url = repo_url.split("?")[0].rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    parts = repo_url.replace("https://github.com/", "").split("/")
    if len(parts) < 2:
        return None, "Invalid repo URL"
    owner, repo = parts[0], parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"    

    def fetch_files(url):
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()

    def find_file_recursive(url):
        items = fetch_files(url)
        if not items:
            return None
        for f in items:
            if f.get("type") == "file" and (f["name"].endswith(".py") or f["name"].endswith(".ipynb")):
                return f["download_url"]
        for f in items:
            if f.get("type") == "dir":
                found = find_file_recursive(f["url"])
                if found:
                    return found
        return None

    download_url = find_file_recursive(api_url)
    if not download_url:
        return None, "No .py or .ipynb files found"
    rr = requests.get(download_url, headers=headers, timeout=15)
    if rr.status_code == 200:
        return rr.text, None
    else:
        return None, f"Failed to download file: {rr.status_code}"
    

def fetch_code_from_link(link, github_token=None):
    """
    Fetch code based on link type:
    - Direct file link (.py or .ipynb) → download directly
    - Link to folder in repo → find first .py/.ipynb in that folder
    - Link to repo → fallback to recursive search in repo
    """
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    # 🔹 1. لو اللينك مباشر لفايل
    if link.endswith(".py") or link.endswith(".ipynb"):
        try:
            r = requests.get(link, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.text, None
            else:
                return None, f"HTTP {r.status_code} when fetching file"
        except Exception as e:
            return None, f"Error fetching direct file: {e}"

    # 🔹 2. لو اللينك فولدر جوه الريبو
    if "/tree/" in link:
        try:
            # Example: https://github.com/user/repo/tree/main/folder
            parts = link.split("github.com/")[1].split("/")
            user, repo = parts[0], parts[1]
            branch = parts[3]
            folder_path = "/".join(parts[4:])
            api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder_path}?ref={branch}"

            r = requests.get(api_url, headers=headers, timeout=15)
            if r.status_code != 200:
                return None, f"Failed to fetch folder content: {r.status_code}"

            items = r.json()
            for f in items:
                if f.get("type") == "file" and (f["name"].endswith(".py") or f["name"].endswith(".ipynb")):
                    file_url = f["download_url"]
                    fr = requests.get(file_url, headers=headers, timeout=15)
                    if fr.status_code == 200:
                        return fr.text, None
                    else:
                        return None, f"Failed to download file from folder: {fr.status_code}"
            return None, "No .py or .ipynb files found in folder"
        except Exception as e:
            return None, f"Error fetching folder: {e}"

    # 🔹 3. لو اللينك ريبو → fallback للقديم
    return fetch_first_code_file_from_repo(link, github_token)



# ---------- LLM grader (Gemini / OpenAI) with mock fallback ----------
def llm_grade_code(code_text, requirements_list):
    """
    Try to call Gemini (if configured) or OpenAI; if quota/problem, return mock result.
    Expects AI_Agent-like structured response (SCORE:, GRADE:, etc.)
    """
    prompt_req = "\n".join(f"- {r}" for r in requirements_list)
    prompt = f"""You are an expert programming instructor. Grade the student's Python solution and produce a structured report.

ASSIGNMENT REQUIREMENTS:
{prompt_req}

STUDENT CODE:
{code_text}

Output EXACTLY in this format (JSON):
{{
  "SCORE": number,
  "GRADE": "A/B/C/D/F",
  "CORRECTNESS": "score/40 - short explanation",
  "CODE_QUALITY": "score/25 - short explanation",
  "COMPLETENESS": "score/20 - short explanation",
  "EFFICIENCY": "score/15 - short explanation",
  "FEEDBACK": "detailed feedback",
  "SUGGESTIONS": "specific improvements",
  "STRENGTHS": "what was good",
  "WEAKNESSES": "what was missing"
}}
Make the JSON valid only (no extra prose).
"""

    # Gemini
    if GEMINI_KEY and HAS_GEMINI:
        try:
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.0))
            text = resp.text
            return try_parse_json(text)
        except Exception as e:
            # fallback to mock
            return mock_grade(code_text, requirements_list, reason=str(e))

    # OpenAI
    if OPENAI_KEY and HAS_OPENAI:
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            text = resp.choices[0].message.content
            return try_parse_json(text)
        except Exception as e:
            return mock_grade(code_text, requirements_list, reason=str(e))

    # No LLM configured -> mock
    return mock_grade(code_text, requirements_list, reason="No LLM API configured (mock)")

def try_parse_json(text):
    # try to find a JSON object in the model output
    try:
        # strip code fences if present
        if text.strip().startswith("```"):
            # remove first and last ```
            text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
        data = json.loads(text)
        return data
    except Exception:
        # try to extract JSON substring
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    # final fallback
    return None

def mock_grade(code_text, requirements_list, reason=None):
    # Simple deterministic mock: mark YES if function names appear, else NO/PARTIAL
    static = analyze_code_statics(code_text)
    defined = static["defined"]
    # map common verbs to function names (simple heuristic)
    heuristics = {
        "add": "add_employee", "view": "view_employees", "update": "update",
        "delete": "delete", "search": "search_employee", "save": "save_to_csv"
    }
    rows = []
    points = []
    for r in requirements_list:
        r_low = r.lower()
        matched = False
        for k, fn in heuristics.items():
            if k in r_low:
                matched = fn in defined
                break
        # decide status
        if matched:
            pts = 1.0
            status = "YES"
        else:
            pts = 0.0
            status = "NO"
        points.append(pts)
        rows.append((r, status))
    avg = sum(points)/len(points) if points else 0
    score = round(100*avg,2)
    grade = "A" if score>=85 else ("B" if score>=70 else ("C" if score>=55 else ("D" if score>=40 else "F")))
    report = {
        "SCORE": score,
        "GRADE": grade,
        "CORRECTNESS": f"{round(score*0.4/100*40,1)}/40 - Mock correctness",
        "CODE_QUALITY": f"{round(score*0.25/100*25,1)}/25 - Mock quality",
        "COMPLETENESS": f"{round(score*0.2/100*20,1)}/20 - Mock completeness",
        "EFFICIENCY": f"{round(score*0.15/100*15,1)}/15 - Mock efficiency",
        "FEEDBACK": f"Mock feedback. Reason: {reason}" if reason else "Mock feedback.",
        "SUGGESTIONS": "Improve testing, edge cases, and add missing functions.",
        "STRENGTHS": "Readable structure where present.",
        "WEAKNESSES": "Missing implementations for some requirements."
    }
    return report

# ---------- Google Classroom OAuth (InstalledAppFlow) ----------
# Note: Streamlit runs on a server — running the flow will open a browser for consent.
SCOPES = [
    "https://www.googleapis.com/auth/classroom.coursework.students",
    "https://www.googleapis.com/auth/classroom.coursework.me",
    "https://www.googleapis.com/auth/classroom.rosters",
    "https://www.googleapis.com/auth/classroom.courses",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/classroom.profile.emails",
    "https://www.googleapis.com/auth/classroom.profile.photos"

]




def get_google_creds(oauth_credentials_path="credentials.json", token_path="token.json"):
    print("=== DEBUG: get_google_creds START ===")
    print(f"Using oauth_credentials_path: {os.path.abspath(oauth_credentials_path)}")
    print(f"Using token_path: {os.path.abspath(token_path)}")
    print(f"Using SCOPES: {SCOPES}")

    creds = None
    if os.path.exists(token_path):
        print("Found existing token.json, trying to load it...")
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        print("Token loaded. Valid:", creds.valid, "Expired:", creds.expired)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(oauth_credentials_path):
                raise FileNotFoundError("credentials.json (OAuth client) not found.")
            print("No valid token found → running InstalledAppFlow...")
            flow = InstalledAppFlow.from_client_secrets_file(oauth_credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # save
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        print("New token.json saved.")

    print("=== DEBUG: get_google_creds END ===")
    return creds


def classroom_service_from_creds(creds):
    return build("classroom", "v1", credentials=creds)

# ---------- Classroom helpers (use OAuth creds) ----------

def list_submissions_for_coursework(service, course_id, coursework_id):
    resp = service.courses().courseWork().studentSubmissions().list(
        courseId=course_id, courseWorkId=coursework_id
    ).execute()
    return resp.get("studentSubmissions", [])

def extract_github_from_submission(sub):
    asm = sub.get("assignmentSubmission", {})
    for att in asm.get("attachments", []):
        link = att.get("link", {}).get("url")
        if link and "github.com" in link:
            return link
    # fallback: check text
    text = asm.get("text", "")
    if "github.com" in text:
        # extract first url
        import re
        m = re.search(r"https?://github\.com[^\s]+", text)
        if m:
            return m.group(0)
    return None





# ---------- Streamlit UI ----------
def render_header():
    st.markdown("""
    <div style="background: linear-gradient(90deg,#f8fafc,#eef2f7);padding:18px;border-radius:10px;">
      <h1 style="text-align:center;color:#0b2545;margin:0;">🤖 AI Auto-Grader</h1>
      <p style="text-align:center;color:#345;opacity:0.8;margin-top:6px;">Auto grade student code from GitHub or uploaded files — integrate with Google Classroom</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    render_header()

    # ---------------- Google Classroom Auth Section ----------------
    st.sidebar.markdown("---")
    st.sidebar.header("🔐 Google Classroom Auth")

    if "svc" not in st.session_state:
        st.session_state["svc"] = None

    oauth_cred_path = st.sidebar.text_input("Path to OAuth credentials.json", value="credentials.json")
    if st.sidebar.button("Authenticate with Google Classroom"):
        try:
            creds = get_google_creds(oauth_credentials_path=oauth_cred_path)
            svc = classroom_service_from_creds(creds)
            st.session_state["svc"] = svc
            st.session_state["creds"] = creds
            st.sidebar.success("✅ Authenticated to Classroom!")
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")

    st.sidebar.title("Controls")
    mode = st.sidebar.radio("Input mode:", ["Manual (upload/edit)", "GitHub link", "From Classroom (OAuth)"])
    st.sidebar.markdown("---")

    # Requirements panel
    st.sidebar.header("Requirements")
    uploaded_req = st.sidebar.file_uploader("Upload Requirements (.docx)", type=["docx"])
    if uploaded_req:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_req.read())
            tmp_path = tmp.name
        reqs = extract_requirements_from_docx(tmp_path)
        st.session_state["requirements"] = reqs
    else:
        if "requirements" not in st.session_state:
            st.session_state["requirements"] = []

    # Show editable requirements in main area
    st.subheader("📝 Requirements (editable)")
    df_reqs = pd.DataFrame({"Requirement": st.session_state["requirements"]})
    edited = st.data_editor(df_reqs, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Requirements"):
        st.session_state["requirements"] = edited["Requirement"].tolist()
        st.success("Requirements saved to session.")
    if st.session_state["requirements"]:
        csv_bytes = pd.DataFrame({"Requirement": st.session_state["requirements"]}).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Requirements (CSV)", data=csv_bytes, file_name="requirements.csv", mime="text/csv")

    st.markdown("---")

    # Code input depending on mode
    code_text = None
    repo_url = None
    github_token = st.sidebar.text_input("GitHub token (optional, for private repos)", type="password")
    if mode == "Manual (upload/edit)":
        st.subheader("Upload student code")
        uploaded_code = st.file_uploader("Upload .py or .ipynb", type=["py","ipynb"])
        if uploaded_code:
            try:
                code_text = code_from_py_or_ipynb_fileobj(uploaded_code)
                st.code(code_text, language="python")
            except Exception as e:
                st.error(f"Failed to load code: {e}")

    elif mode == "GitHub link":
        st.subheader("Provide GitHub repository URL")
        if "repo_url" not in st.session_state:
            st.session_state["repo_url"] = ""
        if "code_text" not in st.session_state:
            st.session_state["code_text"] = None
        if "code_err" not in st.session_state:
            st.session_state["code_err"] = None

        st.subheader("Provide GitHub repository URL")
        st.session_state["repo_url"] = st.text_input(
            "Repository URL (https://github.com/user/repo)",
            value=st.session_state["repo_url"])  

        if st.button("Fetch code from GitHub") and st.session_state["repo_url"]:
            code, err = fetch_code_from_link(st.session_state["repo_url"], github_token if github_token else None)
            st.session_state["code_text"] = code
            st.session_state["code_err"] = err
            

        if st.session_state["code_text"]:
            st.success("Code fetched successfully!")
            st.code(st.session_state["code_text"], language="python")
        elif st.session_state["code_err"]:
            st.error(f"Could not fetch code: {st.session_state['code_err']}")

    else:  # Classroom
        st.subheader("Google Classroom integration")
        st.info("This flow uses OAuth. On first run a browser will open for authentication.")
        oauth_cred_path = st.text_input(
            "Path to OAuth credentials.json (download from Google Cloud)", value="credentials.json"
        )
        classroom_link = st.text_input("Paste assignment URL (Classroom assignment page link)")
        if st.button("Fetch submissions from Classroom") and oauth_cred_path and classroom_link:
            try:
                creds = get_google_creds(oauth_credentials_path=oauth_cred_path)
                service = classroom_service_from_creds(creds)
                # parse classroom link (expected form .../c/COURSE_ID/a/ASSIGNMENT_ID)
                parts = classroom_link.rstrip("/").split("/")
                course_id = decode_classroom_id(parts[-3])
                coursework_id = decode_classroom_id(parts[-1])
                subs = list_submissions_for_coursework(service, course_id, coursework_id)
                st.write(f"Found {len(subs)} submissions.")
                repo_map = {}
                for sub in subs:
                    sid = sub.get("userId")
                    # Get student name
                    try:
                        student = service.userProfiles().get(userId=sid).execute()
                        student_name = student.get("name", {}).get("fullName", f"Student {sid}")
                    except Exception:
                        student_name = f"Student {sid}"

                    repo = extract_github_from_submission(sub)
                    st.write(f"{student_name} -> {repo if repo else 'No repo link found'}")
                    
                    repo_map[sid] = {
                        "repo": repo,  # None if not submitted
                        "submission_id": sub.get("id"),
                        "name": student_name,
                        "email": student.get("emailAddress")}                    
                    
                    st.session_state["classroom_repo_map"] = {
                    "course_id": course_id,
                    "coursework_id": coursework_id,
                    "map": repo_map,
                }
                st.success("Classroom submissions loaded into session.")
            except Exception as e:
                st.error(f"Classroom fetch failed: {e}")


    # ---------- Run grading (single or batch) ----------
    st.markdown("---")
    st.subheader("Run Grading")
    col1, col2 = st.columns([1,3])
    with col1:
        single_mode = st.selectbox("Run mode", ["Single (current code)", "Batch (Classroom submissions)"])
    with col2:
        st.write("Select grading engine and options:")
        use_llm = st.checkbox("Use LLM (Gemini/OpenAI) if available", value=False)
    if st.button("Run Grading Now"):
        reqs = st.session_state.get("requirements", [])
        if single_mode == "Single (current code)":
            if mode == "GitHub link":
                if not st.session_state.get("repo_url"):
                    st.error("Enter a GitHub repo URL first.")
                else:
                    code_text = st.session_state.get("code_text")
                    if not code_text:
                        code_text, err = fetch_code_from_link(st.session_state["repo_url"], github_token if github_token else None)
                    if code_text:
                        st.session_state["code_text"] = code_text
                    else:
                        st.error(f"Could not fetch code: {err}")
            #🔹 For manual upload, code_text is already set from upload
            if not code_text:
                st.error("No code loaded for single-run grading.")
            else:
                static = analyze_code_statics(code_text)
                st.write("Static analysis:", static)
                # LLM or mock
                report = llm_grade_code(code_text, reqs) if use_llm else mock_grade(code_text, reqs, reason="Manual run without LLM")
                display_report(report)
        else:
            # Batch mode: need classroom map
            class_map = st.session_state.get("classroom_repo_map")
            if not class_map:
                st.error("No classroom submissions available — fetch them first.")
            else:
                svc = st.session_state.get("svc")
                if not svc:
                    st.error("⚠️ Please authenticate to Google Classroom first via the sidebar.")
                    return

                results = []
                for sid, info in class_map["map"].items():
                    repo = info["repo"]
                    submission_id = info["submission_id"]
                    student_name = info["name"]
                    student_email = info["email"]

                    # Get student info
                    #student = svc.userProfiles().get(userId=sid).execute()
                    #student_email = student.get("emailAddress")
                    #student_name = student.get("name", {}).get("fullName", "Unknown")

                    st.write(f"📘 Grading {student_name} - {repo} ...")

                    # Get code
                    if repo:
                        code_text, err = fetch_code_from_link(repo, github_token if github_token else None)
                        if not code_text:
                            st.warning(f"⚠️ Could not fetch code for {student_name}: {err}")
                            # Fallback report for failed fetch
                            report = {"SCORE": 0,
                                        "GRADE": "F",
                                        "FEEDBACK": "Could not fetch submission. Score is 0.",
                                        "SUGGESTIONS": "",
                                        "STRENGTHS": "",
                                        "WEAKNESSES": "No submission."}
                        else:
                            if use_llm:
                                report = llm_grade_code(code_text, reqs)
                            else:
                                report = mock_grade(code_text, reqs, reason="Batch mock")
                    else:
                        # Student didn't submit repo
                        report = {"SCORE": 0,
                                    "GRADE": "F",
                                    "FEEDBACK": "You did not submit the assignment and the due date has passed. Score is 0.",
                                    "SUGGESTIONS": "Please submit on time next time.",
                                    "STRENGTHS": "",
                                    "WEAKNESSES": "No submission."}

                    score = report.get("SCORE", 0)

                    # 🔹 إرسال النتيجة بالإيميل
                    if student_email:
                        subject = "Your Assignment Grade"
                        body = f"""
                    Hello {student_name},

                    Your assignment has been graded successfully.

                    📊 Score: {score}
                    📜 Report:
                    {report}

                    Regards,
                    Auto-Grader System
                    """
                        gmail_service = build("gmail", "v1", credentials=st.session_state["creds"])
                        send_email(gmail_service, "me", student_email, subject, body)
                        st.success(f"✅ Sent email to {student_name} ({student_email})")
                    else:
                        st.warning(f"⚠️ No email found for student{student_name}")

                    # Store the grade
                    results.append({
                        "student": student_name,
                        "email": student_email,
                        "score": score,
                        "report": report,
                        "submission_id": submission_id
                    })


                # 🔹 تحويل النتائج لـ DataFrame مع حالة التسليم
                df_res = pd.DataFrame([{
                    "Student Name": r.get("student", "Unknown"),
                    "Email": r.get("email", "N/A"),
                    "Score": r.get("score", 0),
                    "Status": "Submitted" if r.get("report", {}).get("SCORE", 0) > 0 else "Not Submitted"
                } for r in results])

                # 🔹 خلي الترقيم يبدأ من 1
                df_res.index = df_res.index + 1

                # 🔹 عرض الجدول في Streamlit
                st.subheader("📊 Final Results")
                st.dataframe(df_res)

                # 🔹 حفظ CSV
                csv_path = "grading_results.csv"
                df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
                st.success(f"Grades are saved to a file {csv_path}")

                # 🔹 زرار تحميل CSV
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Gareds CSV",
                        data=f,
                        file_name="grading_results.csv",
                        mime="text/csv"
                    )



    st.markdown("---")
    st.write("Notes:")
    st.write("- For Classroom posting and private comments you must use OAuth credentials (teacher account) and consent to scopes.")
    st.write("- LLM calls may be rate-limited; use mock mode for development.")

# ---------- Display helpers ----------
def display_report(report):
    if not report:
        st.error("No report to show.")
        return
    # if it's a string (rare), just print
    if isinstance(report, str):
        st.text(report)
        return
    # show key metrics
    score = report.get("SCORE", "N/A")
    grade = report.get("GRADE", "N/A")
    c1, c2 = st.columns(2)
    c1.metric("Final Score", score)
    c2.metric("Grade", grade)
    st.markdown("### Breakdown")
    for key in ["CORRECTNESS","CODE_QUALITY","COMPLETENESS","EFFICIENCY"]:
        if key in report:
            st.write(f"**{key}:** {report[key]}")
    st.markdown("### Feedback")
    if "FEEDBACK" in report:
        st.info(report["FEEDBACK"])
    if "SUGGESTIONS" in report:
        st.warning(report["SUGGESTIONS"])
    if "STRENGTHS" in report:
        st.success(report["STRENGTHS"])
    if "WEAKNESSES" in report:
        st.error(report["WEAKNESSES"])

def report_to_markdown(report):
    lines = [f"# Auto-Grader Report", f"**Score:** {report.get('SCORE','N/A')}", f"**Grade:** {report.get('GRADE','N/A')}", ""]
    for k in ["CORRECTNESS","CODE_QUALITY","COMPLETENESS","EFFICIENCY"]:
        if k in report:
            lines.append(f"**{k}:** {report[k]}")
    lines.append("\n## Feedback\n")
    lines.append(report.get("FEEDBACK",""))
    lines.append("\n## Suggestions\n")
    lines.append(report.get("SUGGESTIONS",""))
    lines.append("\n## Strengths\n")
    lines.append(report.get("STRENGTHS",""))
    lines.append("\n## Weaknesses\n")
    lines.append(report.get("WEAKNESSES",""))
    return "\n".join(lines)



def decode_classroom_id(encoded_id: str) -> str:
    """
    حاول نفك ترميز ID لو هو Base64. لو مش مظبوط، رجّع النص زي ما هو.
    """
    try:
        decoded = base64.b64decode(encoded_id).decode("utf-8")
        # لو اللي طلع كله أرقام نرجّعه
        if decoded.isdigit():
            return decoded
    except Exception:
        pass
    return encoded_id


if __name__ == "__main__":
    main()
