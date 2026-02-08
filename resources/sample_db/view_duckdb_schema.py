"""
DuckDB 데이터베이스의 COMMENT 조회
View COMMENT from DuckDB database
"""

import os

import duckdb

db_path = os.path.join(os.path.dirname(__file__), "KFTC_sample_table_schemas.duckdb")

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    print("Please run setup_test_db_duckdb.py first")
    exit(1)

conn = duckdb.connect(db_path, read_only=True)

print("=" * 120)
print("금융결제원 금융 이체 정보 - 컬럼 스키마 (COMMENT 포함)")
print("KFTC Financial Transfer Information - Column Schema (with COMMENT)")
print("=" * 120)

# 모든 테이블 목록
tables = [
    "HF_TRNS_TRAN", "CD_TRNS_TRAN", "OB_INQR_TRAN", "OB_TRNS_TRAN",
    "PI_WD_LEDG", "GR_JC_TRAN", "GR_NJ_TRAN", "CMS_REQ_TRAN", "CMS_RES_TRAN"
]

table_descriptions = {
    "HF_TRNS_TRAN": "홈·펌뱅킹 이체 정보",
    "CD_TRNS_TRAN": "CD/ATM 타행 출금 및 이체 정보",
    "OB_INQR_TRAN": "오픈뱅킹 조회 정보",
    "OB_TRNS_TRAN": "오픈뱅킹 이체 정보",
    "PI_WD_LEDG": "출금이체 원장 정보",
    "GR_JC_TRAN": "자동이체 출금청구 내역",
    "GR_NJ_TRAN": "납부자 자동이체 내역",
    "CMS_REQ_TRAN": "CMS 출금 의뢰 내역",
    "CMS_RES_TRAN": "CMS 출금 결과 내역"
}

# 각 테이블별 스키마 출력
for table_name in tables:
    print(f"\n{'=' * 120}")
    print(f"[Table: {table_name}] - {table_descriptions[table_name]}")
    print(f"{'=' * 120}")

    # DuckDB에서 컬럼 정보 및 COMMENT 조회
    result = conn.execute(
        """
        SELECT column_name,
               data_type,
               is_nullable,
               column_comment
        FROM information_schema.columns
        WHERE table_name = ?
        ORDER BY ordinal_position
        """, [table_name]
    ).fetchall()

    if not result:
        print(f"[WARNING] Table {table_name} not found or has no columns")
        continue

    print(f"\n{'No':<4} {'Column Name':<30} {'Data Type':<15} {'Nullable':<10}")
    print("-" * 120)

    for i, (col_name, data_type, nullable, comment) in enumerate(result, 1):
        print(f"{i:<4} {col_name:<30} {data_type:<15} {nullable:<10}")
        if comment:
            # Wrap long comments
            comment_lines = [comment[j: j + 100] for j in range(0, len(comment), 100)]
            for line in comment_lines:
                print(f"     [COMMENT] {line}")
        else:
            print(f"     [COMMENT] (no comment)")
        print()

    # 샘플 데이터 조회 (테이블별로 적절한 컬럼 선택)
    print(f"\n[Sample Data from {table_name} - First 3 Records]")
    print("-" * 120)

    # 각 테이블에 공통적으로 있는 컬럼들을 기준으로 샘플 데이터 조회
    if table_name in ["HF_TRNS_TRAN", "CD_TRNS_TRAN"]:
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, TRNS_DTIME, DPSTR_NM, WD_MN
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'TRNS_DTIME':<18} {'DPSTR_NM':<20} {'WD_MN':>15}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<18} {row[2]:<20} KRW {row[3]:>11,}")

    elif table_name == "OB_INQR_TRAN":
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, INQR_DTIME, INQR_CL_CD, BLNC
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'INQR_DTIME':<18} {'INQR_CL_CD':<15} {'BLNC':>15}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<18} {row[2]:<15} KRW {row[3]:>11,}")

    elif table_name in ["OB_TRNS_TRAN", "PI_WD_LEDG"]:
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, TRNS_DTIME, DPSTR_NM, WD_MN
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'TRNS_DTIME':<18} {'DPSTR_NM':<20} {'WD_MN':>15}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<18} {row[2]:<20} KRW {row[3]:>11,}")

    elif table_name in ["GR_JC_TRAN", "GR_NJ_TRAN"]:
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, WD_DT, DPSTR_NM, WD_MN
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'WD_DT':<12} {'DPSTR_NM':<20} {'WD_MN':>15}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:<20} KRW {row[3]:>11,}")

    elif table_name == "CMS_REQ_TRAN":
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, WD_DT, CMS_NO, WD_MN
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'WD_DT':<12} {'CMS_NO':<20} {'WD_MN':>15}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:<20} KRW {row[3]:>11,}")

    elif table_name == "CMS_RES_TRAN":
        sample_data = conn.execute(
            f"""
            SELECT STD_DT, WD_DT, CMS_NO, WD_MN, WD_RSLT_CD
            FROM {table_name} LIMIT 3
        """
        ).fetchall()
        print(f"{'STD_DT':<12} {'WD_DT':<12} {'CMS_NO':<20} {'WD_MN':>15} {'WD_RSLT_CD':<12}")
        print("-" * 120)
        for row in sample_data:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:<20} KRW {row[3]:>11,} {row[4]:<12}")

print("\n" + "=" * 120)

# PII 컬럼 강조 표시 (모든 테이블의 PII 컬럼)
print("\n[PII Columns (Sensitive Data) - All Tables]")
print("=" * 120)

# 주요 PII 컬럼 목록
pii_columns_all = {
    "DPSTR_NM": "입금자명 - 개인정보",
    "WD_ACNO": "출금계좌번호 - 금융정보",
    "DPS_ACNO": "입금계좌번호 - 금융정보",
    "CUST_ID": "고객ID - 고유식별정보",
    "CRNO": "법인등록번호 - 고유식별정보",
    "CI": "연계정보 - 개인식별정보",
    "FIN_INTT_SVC_NO": "금융이체서비스번호 - 핀테크이용번호",
    "USER_SEQ_NO": "이용자일련번호 - 개인식별정보"
}

for table_name in tables:
    pii_found = conn.execute(
        f"""
        SELECT column_name, data_type, column_comment
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    ).fetchall()

    table_pii = [(col, dtype, cmt) for col, dtype, cmt in pii_found
                 if col in pii_columns_all]

    if table_pii:
        print(f"\n[{table_name}] - {table_descriptions[table_name]}")
        print("-" * 120)
        for col_name, data_type, comment in table_pii:
            print(f"  [PII] {col_name:<25} ({data_type:<15}) - {pii_columns_all.get(col_name, 'PII')}")
            if comment:
                print(f"        Comment: {comment}")

print("\n" + "=" * 120)

# PseuDRAGON이 사용할 스키마 정보
print("\n[Enhanced Schema for PseuDRAGON]")
print("PseuDRAGON uses both column names and COMMENT for accurate PII detection")
print("-" * 120)

# Enhanced schema 예시
print("\nEnhanced Schema Format:")
print("{\n  'column_name': {")
print("    'type': 'VARCHAR',")
print("    'comment': 'Column description',")
print("    'nullable': True,")
print("    'sample_value': 'example'")
print("  }\n}")

print("\nExample for 'DPSTR_NM' from HF_TRNS_TRAN:")
dpstr_info = conn.execute(
    """
    SELECT column_name,
           data_type,
           is_nullable,
           column_comment
    FROM information_schema.columns
    WHERE table_name = 'HF_TRNS_TRAN'
      AND column_name = 'DPSTR_NM'
    """
).fetchone()

if dpstr_info:
    sample_value = conn.execute("SELECT DPSTR_NM FROM HF_TRNS_TRAN LIMIT 1").fetchone()[0]

    print("{")
    print(f"  'DPSTR_NM': {{")
    print(f"    'type': '{dpstr_info[1]}',")
    print(f"    'nullable': {dpstr_info[2] == 'YES'},")
    print(f"    'comment': '{dpstr_info[3]}',")
    print(f"    'sample_value': '{sample_value}'")
    print("  }")
    print("}")

print("\n" + "=" * 120)

print("\n[Summary]")
print(f"Total Tables: {len(tables)}")
for table_name in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  - {table_name}: {count:,} records - {table_descriptions[table_name]}")

print("\n[OK] DuckDB supports COMMENT on columns.")
print("[OK] PseuDRAGON queries both column names and COMMENT")
print("[OK] This enables accurate PII identification in Korean enterprise databases.")
print("\nQuery to get COMMENT:")
print(
    """
    SELECT column_name, column_comment
    FROM information_schema.columns
    WHERE table_name = 'HF_TRNS_TRAN'
    """
)

conn.close()
