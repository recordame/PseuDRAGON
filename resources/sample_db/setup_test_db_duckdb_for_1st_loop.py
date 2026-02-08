"""
테스트 데이터베이스 생성 (DuckDB)
Test Database Setup (DuckDB)

DuckDB를 사용하여 COMMENT를 지원하는 데이터베이스 생성
Using DuckDB for COMMENT support in database schema
"""

import os
import random
from datetime import datetime, timedelta

try:
    import duckdb
except ImportError:
    print("DuckDB is not installed. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "duckdb"])
    import duckdb

# 한국 성씨 및 이름 데이터
KOREAN_SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍", ]
KOREAN_GIVEN_NAMES_1 = ["민", "서", "지", "하", "예", "도", "수", "은", "현", "준", "우", "진", "영", "재", "성", "태", "승", "경", "동", "철", ]
KOREAN_GIVEN_NAMES_2 = ["준", "윤", "호", "우", "진", "민", "서", "영", "수", "현", "아", "연", "희", "정", "미", "선", "혁", "석", "훈", "빈", ]

# 은행 코드 (실제 금융결제원 표준 은행 코드)
BANK_CODES = {
    "004": "KB국민은행",
    "088": "신한은행",
    "020": "우리은행",
    "081": "하나은행",
    "003": "IBK기업은행",
    "011": "NH농협은행",
    "023": "SC제일은행",
    "027": "한국씨티은행",
    "031": "대구은행",
    "032": "부산은행",
    "034": "광주은행",
    "035": "제주은행",
    "037": "전북은행",
    "039": "경남은행",
    "045": "새마을금고",
    "048": "신협",
    "050": "상호저축은행",
    "071": "우체국",
    "089": "케이뱅크",
    "090": "카카오뱅크",
    "092": "토스뱅크",
}


def generate_korean_name():
    """한국 이름 생성"""
    return (random.choice(KOREAN_SURNAMES) + random.choice(KOREAN_GIVEN_NAMES_1) + random.choice(KOREAN_GIVEN_NAMES_2))


def generate_bank_tran_id():
    """은행거래고유번호 생성 (AN 20자)"""
    prefix = "F"
    random_part = "".join([str(random.randint(0, 9)) for _ in range(9)])
    suffix = "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(10)])
    return f"{prefix}{random_part}{suffix}"


def generate_fintech_use_num():
    """핀테크이용번호 생성 (AN 24자)"""
    return "".join([str(random.randint(0, 9)) for _ in range(24)])


def generate_account_num():
    """계좌번호 생성 (N 16자)"""
    return "".join([str(random.randint(0, 9)) for _ in range(16)])


def generate_business_num():
    """사업자등록번호 생성 (N 10자)"""
    return "".join([str(random.randint(0, 9)) for _ in range(10)])


def generate_datetime(days_ago_max=365):
    """랜덤 날짜/시간 생성"""
    days_ago = random.randint(0, days_ago_max)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    dt = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes, seconds=seconds)
    return dt.strftime("%Y%m%d%H%M%S")


def initialize_sample_database(db_path: str, num_records: int = 1000):
    """
    금융결제원 금융 이체 정보 데이터베이스 초기화 (DuckDB)
    Initialize database based on KFTC Financial Transfer Information (DuckDB)

    9개 테이블: HF_TRNS_TRAN, CD_TRNS_TRAN, OB_INQR_TRAN, OB_TRNS_TRAN, PI_WD_LEDG,
               GR_JC_TRAN, GR_NJ_TRAN, CMS_REQ_TRAN, CMS_RES_TRAN

    Args:
        db_path: 데이터베이스 파일 경로
        num_records: 생성할 레코드 수 (각 테이블당)
    """

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
        except PermissionError:
            print(f"Warning: Could not remove existing DB. Using existing file.")

    # DuckDB 연결
    conn = duckdb.connect(db_path)

    print("=" * 80)
    print("금융결제원 금융 이체 정보 데이터베이스 생성 (DuckDB)")
    print("KFTC Financial Transfer Information Database Setup (DuckDB)")
    print("=" * 80)

    # 1. HF_TRNS_TRAN (Home/Firm Banking Transfer Information)
    print("\n[1/9] Creating 'HF_TRNS_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE HF_TRNS_TRAN
        (
            STD_DT          NVARCHAR(8) NOT NULL,
            TRNS_DTIME      NVARCHAR(14) NOT NULL,
            RCMS_BSWR_CL_CD NVARCHAR(2),
            TRNS_MED_CL_CD  NVARCHAR(2),
            TRNS_CL_CD      NVARCHAR(2),
            CRNO            NVARCHAR(13),
            CUST_ID         NVARCHAR(50),
            WD_BANK_CD      NVARCHAR(3),
            WD_ACNO         NVARCHAR(20),
            DPSTR_NM        NVARCHAR(50),
            DPS_BANK_CD     NVARCHAR(3),
            DPS_ACNO        NVARCHAR(20),
            WD_MN           BIGINT,
            FEE             BIGINT
        )
        """
    )

    hf_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("TRNS_DTIME", "Transaction datetime(YYYYMMDDhhmmss)"),
        ("RCMS_BSWR_CL_CD", "Recipient inquiry classification code(01:Inquired, 02:Not inquired, 99:No information)"),
        ("TRNS_MED_CL_CD", "Transaction medium classification code(11:Internet banking, 12:Mobile banking, 13:Phone banking, 99:No information)"),
        ("TRNS_CL_CD", "Transaction classification code(01:Transfer, 02:Utility payment, 03:Firm banking)"),
        ("CRNO", "Corporate registration number"),
        ("CUST_ID", "Customer ID"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("DPSTR_NM", "Depositor name"),
        ("DPS_BANK_CD", "Deposit bank code"),
        ("DPS_ACNO", "Deposit account number"),
        ("WD_MN", "Withdrawal amount"),
        ("FEE", "Fee")
    ]
    for col, comment in hf_comments:
        conn.execute(f"COMMENT ON COLUMN HF_TRNS_TRAN.{col} IS '{comment}'")

    # 2. CD_TRNS_TRAN (CD/ATM Inter-bank Withdrawal and Transfer Information)
    print("[2/9] Creating 'CD_TRNS_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE CD_TRNS_TRAN
        (
            STD_DT          NVARCHAR(8) NOT NULL,
            TRNS_DTIME      NVARCHAR(14) NOT NULL,
            RCMS_BSWR_CL_CD NVARCHAR(2),
            TRNS_MED_CL_CD  NVARCHAR(2),
            TRNS_CL_CD      NVARCHAR(2),
            CUST_ID         NVARCHAR(50),
            WD_BANK_CD      NVARCHAR(3),
            WD_ACNO         NVARCHAR(20),
            DPSTR_NM        NVARCHAR(50),
            DPS_BANK_CD     NVARCHAR(3),
            DPS_ACNO        NVARCHAR(20),
            WD_MN           BIGINT,
            FEE             BIGINT
        )
        """
    )

    cd_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("TRNS_DTIME", "Transaction datetime(YYYYMMDDhhmmss)"),
        ("RCMS_BSWR_CL_CD", "Recipient inquiry classification code(01:Inquired, 02:Not inquired, 99:No information)"),
        ("TRNS_MED_CL_CD", "Transaction medium classification code(21:CD/ATM)"),
        ("TRNS_CL_CD", "Transaction classification code(01:Transfer, 04:Inter-bank withdrawal)"),
        ("CUST_ID", "Customer ID"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("DPSTR_NM", "Depositor name"),
        ("DPS_BANK_CD", "Deposit bank code"),
        ("DPS_ACNO", "Deposit account number"),
        ("WD_MN", "Withdrawal amount"),
        ("FEE", "Fee")
    ]
    for col, comment in cd_comments:
        conn.execute(f"COMMENT ON COLUMN CD_TRNS_TRAN.{col} IS '{comment}'")

    # 3. OB_INQR_TRAN (Open Banking Inquiry Information)
    print("[3/9] Creating 'OB_INQR_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE OB_INQR_TRAN
        (
            STD_DT          NVARCHAR(8) NOT NULL,
            INQR_DTIME      NVARCHAR(14) NOT NULL,
            INQR_CL_CD      NVARCHAR(2),
            USER_SEQ_NO     NVARCHAR(10),
            CI              NVARCHAR(88),
            FIN_INTT_SVC_NO NVARCHAR(24),
            BLNC            BIGINT
        )
        """
    )

    ob_inqr_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("INQR_DTIME", "Inquiry datetime(YYYYMMDDhhmmss)"),
        ("INQR_CL_CD", "Inquiry classification code(01:Balance inquiry, 02:Transaction history inquiry)"),
        ("USER_SEQ_NO", "User sequence number"),
        ("CI", "CI - Connecting Information, identification value issued by identity verification agency"),
        ("FIN_INTT_SVC_NO", "Financial transfer service number"),
        ("BLNC", "Balance")
    ]
    for col, comment in ob_inqr_comments:
        conn.execute(f"COMMENT ON COLUMN OB_INQR_TRAN.{col} IS '{comment}'")

    # 4. OB_TRNS_TRAN (Open Banking Transfer Information)
    print("[4/9] Creating 'OB_TRNS_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE OB_TRNS_TRAN
        (
            STD_DT          NVARCHAR(8) NOT NULL,
            TRNS_DTIME      NVARCHAR(14) NOT NULL,
            RCMS_BSWR_CL_CD NVARCHAR(2),
            USER_SEQ_NO     NVARCHAR(10),
            CI              NVARCHAR(88),
            WD_BANK_CD      NVARCHAR(3),
            FIN_INTT_SVC_NO NVARCHAR(24),
            DPSTR_NM        NVARCHAR(50),
            DPS_BANK_CD     NVARCHAR(3),
            DPS_ACNO        NVARCHAR(20),
            WD_MN           BIGINT
        )
        """
    )

    ob_trns_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("TRNS_DTIME", "Transaction datetime(YYYYMMDDhhmmss)"),
        ("RCMS_BSWR_CL_CD", "Recipient inquiry classification code(01:Inquired, 02:Not inquired, 99:No information)"),
        ("USER_SEQ_NO", "User sequence number"),
        ("CI", "CI - Connecting Information, identification value issued by identity verification agency"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("FIN_INTT_SVC_NO", "Financial transfer service number"),
        ("DPSTR_NM", "Depositor name"),
        ("DPS_BANK_CD", "Deposit bank code"),
        ("DPS_ACNO", "Deposit account number"),
        ("WD_MN", "Withdrawal amount")
    ]
    for col, comment in ob_trns_comments:
        conn.execute(f"COMMENT ON COLUMN OB_TRNS_TRAN.{col} IS '{comment}'")

    # 5. PI_WD_LEDG (Withdrawal Transfer Ledger Information)
    print("[5/9] Creating 'PI_WD_LEDG' table...")
    conn.execute(
        """
        CREATE TABLE PI_WD_LEDG
        (
            STD_DT          NVARCHAR(8) NOT NULL,
            TRNS_DTIME      NVARCHAR(14) NOT NULL,
            RCMS_BSWR_CL_CD NVARCHAR(2),
            CMS_NO          NVARCHAR(13),
            USER_SEQ_NO     NVARCHAR(10),
            CI              NVARCHAR(88),
            WD_BANK_CD      NVARCHAR(3),
            WD_ACNO         NVARCHAR(20),
            DPSTR_NM        NVARCHAR(50),
            DPS_BANK_CD     NVARCHAR(3),
            DPS_ACNO        NVARCHAR(20),
            WD_MN           BIGINT
        )
        """
    )

    pi_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("TRNS_DTIME", "Transaction datetime(YYYYMMDDhhmmss)"),
        ("RCMS_BSWR_CL_CD", "Recipient inquiry classification code(01:Inquired, 02:Not inquired, 99:No information)"),
        ("CMS_NO", "CMS number"),
        ("USER_SEQ_NO", "User sequence number"),
        ("CI", "CI - Connecting Information, identification value issued by identity verification agency"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("DPSTR_NM", "Depositor name"),
        ("DPS_BANK_CD", "Deposit bank code"),
        ("DPS_ACNO", "Deposit account number"),
        ("WD_MN", "Withdrawal amount")
    ]
    for col, comment in pi_comments:
        conn.execute(f"COMMENT ON COLUMN PI_WD_LEDG.{col} IS '{comment}'")

    # 6. GR_JC_TRAN (Auto Transfer Withdrawal Request History)
    print("[6/9] Creating 'GR_JC_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE GR_JC_TRAN
        (
            STD_DT     NVARCHAR(8) NOT NULL,
            REQ_DT     NVARCHAR(8) NOT NULL,
            WD_DT      NVARCHAR(8),
            CMS_NO     NVARCHAR(13),
            CRNO       NVARCHAR(13),
            DPSTR_NM   NVARCHAR(50),
            WD_BANK_CD NVARCHAR(3),
            WD_ACNO    NVARCHAR(20),
            WD_MN      BIGINT,
            DEALCO_CD  NVARCHAR(3)
        )
        """
    )

    gr_jc_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("REQ_DT", "Request date(YYYYMMDD)"),
        ("WD_DT", "Withdrawal date(YYYYMMDD)"),
        ("CMS_NO", "CMS number"),
        ("CRNO", "Corporate registration number"),
        ("DPSTR_NM", "Depositor name"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("WD_MN", "Withdrawal amount"),
        ("DEALCO_CD", "Handling institution code")
    ]
    for col, comment in gr_jc_comments:
        conn.execute(f"COMMENT ON COLUMN GR_JC_TRAN.{col} IS '{comment}'")

    # 7. GR_NJ_TRAN (Payer Auto Transfer History)
    print("[7/9] Creating 'GR_NJ_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE GR_NJ_TRAN
        (
            STD_DT     NVARCHAR(8) NOT NULL,
            TRNS_DTIME NVARCHAR(14) NOT NULL,
            WD_DT      NVARCHAR(8),
            CMS_NO     NVARCHAR(13),
            CRNO       NVARCHAR(13),
            DPSTR_NM   NVARCHAR(50),
            WD_BANK_CD NVARCHAR(3),
            WD_ACNO    NVARCHAR(20),
            WD_MN      BIGINT,
            DEALCO_CD  NVARCHAR(3)
        )
        """
    )

    gr_nj_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("TRNS_DTIME", "Transaction datetime(YYYYMMDDhhmmss)"),
        ("WD_DT", "Withdrawal date(YYYYMMDD)"),
        ("CMS_NO", "CMS number"),
        ("CRNO", "Corporate registration number"),
        ("DPSTR_NM", "Depositor name"),
        ("WD_BANK_CD", "Withdrawal bank code "),
        ("WD_ACNO", "Withdrawal account number"),
        ("WD_MN", "Withdrawal amount"),
        ("DEALCO_CD", "Handling institution code")
    ]
    for col, comment in gr_nj_comments:
        conn.execute(f"COMMENT ON COLUMN GR_NJ_TRAN.{col} IS '{comment}'")

    # 8. CMS_REQ_TRAN (CMS Withdrawal Request History)
    print("[8/9] Creating 'CMS_REQ_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE CMS_REQ_TRAN
        (
            STD_DT     NVARCHAR(8) NOT NULL,
            WD_DT      NVARCHAR(8) NOT NULL,
            CMS_NO     NVARCHAR(13),
            CRNO       NVARCHAR(13),
            WD_BANK_CD NVARCHAR(3),
            WD_ACNO    NVARCHAR(20),
            WD_MN      BIGINT
        )
        """
    )

    cms_req_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("WD_DT", "Withdrawal date(Scheduled withdrawal date)"),
        ("CMS_NO", "CMS number"),
        ("CRNO", "Corporate registration number"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("WD_MN", "Withdrawal amount")
    ]
    for col, comment in cms_req_comments:
        conn.execute(f"COMMENT ON COLUMN CMS_REQ_TRAN.{col} IS '{comment}'")

    # 9. CMS_RES_TRAN (CMS Withdrawal Result History)
    print("[9/9] Creating 'CMS_RES_TRAN' table...")
    conn.execute(
        """
        CREATE TABLE CMS_RES_TRAN
        (
            STD_DT     NVARCHAR(8) NOT NULL,
            WD_DT      NVARCHAR(8) NOT NULL,
            CMS_NO     NVARCHAR(13),
            CRNO       NVARCHAR(13),
            WD_BANK_CD NVARCHAR(3),
            WD_ACNO    NVARCHAR(20),
            WD_MN      BIGINT,
            WD_RSLT_CD NVARCHAR(2)
        )
        """
    )

    cms_res_comments = [
        ("STD_DT", "Standard date(YYYYMMDD)"),
        ("WD_DT", "Withdrawal date(Withdrawal processing date)"),
        ("CMS_NO", "CMS number"),
        ("CRNO", "Corporate registration number"),
        ("WD_BANK_CD", "Withdrawal bank code"),
        ("WD_ACNO", "Withdrawal account number"),
        ("WD_MN", "Withdrawal amount"),
        ("WD_RSLT_CD", "Withdrawal result code(00:Normal, 01:Insufficient balance, 99:Other)")
    ]
    for col, comment in cms_res_comments:
        conn.execute(f"COMMENT ON COLUMN CMS_RES_TRAN.{col} IS '{comment}'")

    print("\n[INDEX] Creating indexes for all tables...")
    # HF_TRNS_TRAN indexes
    conn.execute("CREATE INDEX idx_hf_trns_dtime ON HF_TRNS_TRAN(TRNS_DTIME)")
    conn.execute("CREATE INDEX idx_hf_dpstr_nm ON HF_TRNS_TRAN(DPSTR_NM)")
    conn.execute("CREATE INDEX idx_hf_wd_acno ON HF_TRNS_TRAN(WD_ACNO)")

    # CD_TRNS_TRAN indexes
    conn.execute("CREATE INDEX idx_cd_trns_dtime ON CD_TRNS_TRAN(TRNS_DTIME)")
    conn.execute("CREATE INDEX idx_cd_dpstr_nm ON CD_TRNS_TRAN(DPSTR_NM)")

    # OB_INQR_TRAN indexes
    conn.execute("CREATE INDEX idx_ob_inqr_dtime ON OB_INQR_TRAN(INQR_DTIME)")
    conn.execute("CREATE INDEX idx_ob_inqr_ci ON OB_INQR_TRAN(CI)")

    # OB_TRNS_TRAN indexes
    conn.execute("CREATE INDEX idx_ob_trns_dtime ON OB_TRNS_TRAN(TRNS_DTIME)")
    conn.execute("CREATE INDEX idx_ob_trns_ci ON OB_TRNS_TRAN(CI)")

    # PI_WD_LEDG indexes
    conn.execute("CREATE INDEX idx_pi_trns_dtime ON PI_WD_LEDG(TRNS_DTIME)")
    conn.execute("CREATE INDEX idx_pi_ci ON PI_WD_LEDG(CI)")

    # GR_JC_TRAN indexes
    conn.execute("CREATE INDEX idx_gr_jc_req_dt ON GR_JC_TRAN(REQ_DT)")
    conn.execute("CREATE INDEX idx_gr_jc_cms_no ON GR_JC_TRAN(CMS_NO)")

    # GR_NJ_TRAN indexes
    conn.execute("CREATE INDEX idx_gr_nj_trns_dtime ON GR_NJ_TRAN(TRNS_DTIME)")
    conn.execute("CREATE INDEX idx_gr_nj_cms_no ON GR_NJ_TRAN(CMS_NO)")

    # CMS_REQ_TRAN indexes
    conn.execute("CREATE INDEX idx_cms_req_wd_dt ON CMS_REQ_TRAN(WD_DT)")
    conn.execute("CREATE INDEX idx_cms_req_cms_no ON CMS_REQ_TRAN(CMS_NO)")

    # CMS_RES_TRAN indexes
    conn.execute("CREATE INDEX idx_cms_res_wd_dt ON CMS_RES_TRAN(WD_DT)")
    conn.execute("CREATE INDEX idx_cms_res_cms_no ON CMS_RES_TRAN(CMS_NO)")

    print(f"\n[DATA] Generating {num_records} sample records for each table...")
    print("      Using Korean names and realistic financial data")

    # Generate sample data for HF_TRNS_TRAN
    print("\n   Generating HF_TRNS_TRAN records...")
    hf_records = []
    for i in range(num_records):
        record = (
            datetime.now().strftime("%Y%m%d"),  # STD_DT
            generate_datetime(),  # TRNS_DTIME
            random.choice(["01", "02", "99"]),  # RCMS_BSWR_CL_CD
            random.choice(["11", "12", "13", "99"]),  # TRNS_MED_CL_CD
            random.choice(["01", "02", "03"]),  # TRNS_CL_CD
            generate_business_num(),  # CRNO
            f"CUST{random.randint(1000000, 9999999)}",  # CUST_ID
            random.choice(list(BANK_CODES.keys())),  # WD_BANK_CD
            generate_account_num(),  # WD_ACNO
            generate_korean_name(),  # DPSTR_NM
            random.choice(list(BANK_CODES.keys())),  # DPS_BANK_CD
            generate_account_num(),  # DPS_ACNO
            int(random.uniform(10000, 5000000)),  # WD_MN
            int(random.uniform(0, 1000))  # FEE
        )
        hf_records.append(record)

    conn.executemany(
        """
        INSERT INTO HF_TRNS_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hf_records
    )

    # Generate sample data for CD_TRNS_TRAN
    print("   Generating CD_TRNS_TRAN records...")
    cd_records = []
    for i in range(num_records):
        record = (
            datetime.now().strftime("%Y%m%d"),
            generate_datetime(),
            random.choice(["01", "02", "99"]),
            "21",  # CD/ATM
            random.choice(["01", "04"]),
            f"CUST{random.randint(1000000, 9999999)}",
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            generate_korean_name(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 2000000)),
            int(random.uniform(0, 1000))
        )
        cd_records.append(record)

    conn.executemany(
        """
        INSERT INTO CD_TRNS_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, cd_records
    )

    # Generate sample data for OB_INQR_TRAN
    print("   Generating OB_INQR_TRAN records...")
    ob_inqr_records = []
    for i in range(num_records):
        record = (
            datetime.now().strftime("%Y%m%d"),
            generate_datetime(),
            random.choice(["01", "02"]),
            f"{random.randint(1000000000, 9999999999)}",
            "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/") for _ in range(88)]),
            generate_fintech_use_num(),
            int(random.uniform(100000, 10000000))
        )
        ob_inqr_records.append(record)

    conn.executemany(
        """
        INSERT INTO OB_INQR_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ob_inqr_records
    )

    # Generate sample data for OB_TRNS_TRAN
    print("   Generating OB_TRNS_TRAN records...")
    ob_trns_records = []
    for i in range(num_records):
        record = (
            datetime.now().strftime("%Y%m%d"),
            generate_datetime(),
            random.choice(["01", "02", "99"]),
            f"{random.randint(1000000000, 9999999999)}",
            "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/") for _ in range(88)]),
            random.choice(list(BANK_CODES.keys())),
            generate_fintech_use_num(),
            generate_korean_name(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 5000000))
        )
        ob_trns_records.append(record)

    conn.executemany(
        """
        INSERT INTO OB_TRNS_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ob_trns_records
    )

    # Generate sample data for PI_WD_LEDG
    print("   Generating PI_WD_LEDG records...")
    pi_records = []
    for i in range(num_records):
        record = (
            datetime.now().strftime("%Y%m%d"),
            generate_datetime(),
            random.choice(["01", "02", "99"]),
            f"CMS{random.randint(1000000000, 9999999999)}",
            f"{random.randint(1000000000, 9999999999)}",
            "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/") for _ in range(88)]),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            generate_korean_name(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 3000000))
        )
        pi_records.append(record)

    conn.executemany(
        """
        INSERT INTO PI_WD_LEDG
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, pi_records
    )

    # Generate sample data for GR_JC_TRAN
    print("   Generating GR_JC_TRAN records...")
    gr_jc_records = []
    for i in range(num_records):
        req_dt = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y%m%d")
        wd_dt = (datetime.strptime(req_dt, "%Y%m%d") + timedelta(days=random.randint(0, 5))).strftime("%Y%m%d")
        record = (
            datetime.now().strftime("%Y%m%d"),
            req_dt,
            wd_dt,
            f"CMS{random.randint(1000000000, 9999999999)}",
            generate_business_num(),
            generate_korean_name(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 1000000)),
            random.choice(list(BANK_CODES.keys())[:10])
        )
        gr_jc_records.append(record)

    conn.executemany(
        """
        INSERT INTO GR_JC_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, gr_jc_records
    )

    # Generate sample data for GR_NJ_TRAN
    print("   Generating GR_NJ_TRAN records...")
    gr_nj_records = []
    for i in range(num_records):
        wd_dt = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y%m%d")
        record = (
            datetime.now().strftime("%Y%m%d"),
            generate_datetime(),
            wd_dt,
            f"CMS{random.randint(1000000000, 9999999999)}",
            generate_business_num(),
            generate_korean_name(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 1000000)),
            random.choice(list(BANK_CODES.keys())[:10])
        )
        gr_nj_records.append(record)

    conn.executemany(
        """
        INSERT INTO GR_NJ_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, gr_nj_records
    )

    # Generate sample data for CMS_REQ_TRAN
    print("   Generating CMS_REQ_TRAN records...")
    cms_req_records = []
    for i in range(num_records):
        wd_dt = (datetime.now() + timedelta(days=random.randint(0, 10))).strftime("%Y%m%d")
        record = (
            datetime.now().strftime("%Y%m%d"),
            wd_dt,
            f"CMS{random.randint(1000000000, 9999999999)}",
            generate_business_num(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 1000000))
        )
        cms_req_records.append(record)

    conn.executemany(
        """
        INSERT INTO CMS_REQ_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, cms_req_records
    )

    # Generate sample data for CMS_RES_TRAN
    print("   Generating CMS_RES_TRAN records...")
    cms_res_records = []
    for i in range(num_records):
        wd_dt = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y%m%d")
        record = (
            datetime.now().strftime("%Y%m%d"),
            wd_dt,
            f"CMS{random.randint(1000000000, 9999999999)}",
            generate_business_num(),
            random.choice(list(BANK_CODES.keys())),
            generate_account_num(),
            int(random.uniform(10000, 1000000)),
            random.choice(["00", "01", "99"])
        )
        cms_res_records.append(record)

    conn.executemany(
        """
        INSERT INTO CMS_RES_TRAN
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, cms_res_records
    )

    print("\n" + "=" * 80)
    print("Database Setup Complete!")
    print("=" * 80)

    # 통계 출력
    print(f"\n Database Statistics:")
    print(f"   - Database Path: {db_path}")
    print(f"   - Database Type: DuckDB (supports COMMENT)")
    print(f"   - Total Tables: 9")

    tables = [
        "HF_TRNS_TRAN", "CD_TRNS_TRAN", "OB_INQR_TRAN", "OB_TRNS_TRAN",
        "PI_WD_LEDG", "GR_JC_TRAN", "GR_NJ_TRAN", "CMS_REQ_TRAN", "CMS_RES_TRAN"
    ]

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"   - {table}: {count:,} records")

    print(f"\n Sample Records from HF_TRNS_TRAN (first 3):")
    result = conn.execute(
        """
        SELECT TRNS_DTIME, DPSTR_NM, WD_MN, TRNS_MED_CL_CD
        FROM HF_TRNS_TRAN LIMIT 3
        """
    ).fetchall()

    for row in result:
        print(f"   - {row[0]}: {row[1]}, KRW {row[2]:,}, 매체코드: {row[3]}")

    print("\n" + "=" * 80)

    conn.close()


if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), "KFTC_sample_table_schemas.duckdb")
    initialize_sample_database(db_path, num_records=1000)
    print("\n Setup complete! You can now use this DuckDB database for testing PseuDRAGON.")
    print(f"   Database location: {db_path}")
    print("\n Note: DuckDB supports COMMENT on columns, which helps PseuDRAGON")
    print("       identify PII more accurately using both column names and descriptions.")
