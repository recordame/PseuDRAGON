"""Canonical Korean SQL COMMENT values for the 92-column evaluation schema."""

from __future__ import annotations

from pathlib import Path


KOREAN_COLUMN_COMMENTS: dict[str, dict[str, str]] = {
    "HF_TRNS_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "TRNS_DTIME": "거래일시(YYYYMMDDhhmmss)",
        "RCMS_BSWR_CL_CD": "수신 조회 구분 코드(01:조회, 02:미조회, 99:정보없음)",
        "TRNS_MED_CL_CD": "거래 매체 구분 코드(11:인터넷뱅킹, 12:모바일뱅킹, 13:텔레뱅킹, 99:정보없음)",
        "TRNS_CL_CD": "거래 구분 코드(01:이체, 02:공과금납부, 03:펌뱅킹)",
        "CRNO": "법인등록번호",
        "CUST_ID": "고객 ID",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "DPSTR_NM": "입금자명",
        "DPS_BANK_CD": "입금은행코드",
        "DPS_ACNO": "입금계좌번호",
        "WD_MN": "출금금액",
        "FEE": "수수료",
    },
    "CD_TRNS_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "TRNS_DTIME": "거래일시(YYYYMMDDhhmmss)",
        "RCMS_BSWR_CL_CD": "수신 조회 구분 코드(01:조회, 02:미조회, 99:정보없음)",
        "TRNS_MED_CL_CD": "거래 매체 구분 코드(21:CD/ATM)",
        "TRNS_CL_CD": "거래 구분 코드(01:이체, 04:타행출금)",
        "CUST_ID": "고객 ID",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "DPSTR_NM": "입금자명",
        "DPS_BANK_CD": "입금은행코드",
        "DPS_ACNO": "입금계좌번호",
        "WD_MN": "출금금액",
        "FEE": "수수료",
    },
    "OB_INQR_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "INQR_DTIME": "조회일시(YYYYMMDDhhmmss)",
        "INQR_CL_CD": "조회 구분 코드(01:잔액조회, 02:거래내역조회)",
        "USER_SEQ_NO": "사용자일련번호",
        "CI": "연계정보(CI, 본인확인기관이 발급한 식별값)",
        "FIN_INTT_SVC_NO": "금융이체서비스번호",
        "BLNC": "잔액",
    },
    "OB_TRNS_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "TRNS_DTIME": "거래일시(YYYYMMDDhhmmss)",
        "RCMS_BSWR_CL_CD": "수신 조회 구분 코드(01:조회, 02:미조회, 99:정보없음)",
        "USER_SEQ_NO": "사용자일련번호",
        "CI": "연계정보(CI, 본인확인기관이 발급한 식별값)",
        "WD_BANK_CD": "출금은행코드",
        "FIN_INTT_SVC_NO": "금융이체서비스번호",
        "DPSTR_NM": "입금자명",
        "DPS_BANK_CD": "입금은행코드",
        "DPS_ACNO": "입금계좌번호",
        "WD_MN": "출금금액",
    },
    "PI_WD_LEDG": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "TRNS_DTIME": "거래일시(YYYYMMDDhhmmss)",
        "RCMS_BSWR_CL_CD": "수신 조회 구분 코드(01:조회, 02:미조회, 99:정보없음)",
        "CMS_NO": "자금관리서비스(CMS) 번호",
        "USER_SEQ_NO": "사용자일련번호",
        "CI": "연계정보(CI, 본인확인기관이 발급한 식별값)",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "DPSTR_NM": "입금자명",
        "DPS_BANK_CD": "입금은행코드",
        "DPS_ACNO": "입금계좌번호",
        "WD_MN": "출금금액",
    },
    "GR_JC_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "REQ_DT": "요청일자(YYYYMMDD)",
        "WD_DT": "출금일자(YYYYMMDD)",
        "CMS_NO": "자금관리서비스(CMS) 번호",
        "CRNO": "법인등록번호",
        "DPSTR_NM": "입금자명",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "WD_MN": "출금금액",
        "DEALCO_CD": "취급기관코드",
    },
    "GR_NJ_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "TRNS_DTIME": "거래일시(YYYYMMDDhhmmss)",
        "WD_DT": "출금일자(YYYYMMDD)",
        "CMS_NO": "자금관리서비스(CMS) 번호",
        "CRNO": "법인등록번호",
        "DPSTR_NM": "입금자명",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "WD_MN": "출금금액",
        "DEALCO_CD": "취급기관코드",
    },
    "CMS_REQ_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "WD_DT": "출금일자(출금예정일)",
        "CMS_NO": "자금관리서비스(CMS) 번호",
        "CRNO": "법인등록번호",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "WD_MN": "출금금액",
    },
    "CMS_RES_TRAN": {
        "STD_DT": "표준일자(YYYYMMDD)",
        "WD_DT": "출금일자(출금처리일)",
        "CMS_NO": "자금관리서비스(CMS) 번호",
        "CRNO": "법인등록번호",
        "WD_BANK_CD": "출금은행코드",
        "WD_ACNO": "출금계좌번호",
        "WD_MN": "출금금액",
        "WD_RSLT_CD": "출금결과코드(00:정상, 01:잔액부족, 99:기타)",
    },
}


def apply_korean_comments(connection) -> None:
    """Apply and verify all canonical Korean comments on an open DuckDB connection."""
    expected_count = sum(len(columns) for columns in KOREAN_COLUMN_COMMENTS.values())
    if expected_count != 92:
        raise RuntimeError(f"Expected 92 comments, found {expected_count}")

    for table_name, columns in KOREAN_COLUMN_COMMENTS.items():
        existing = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
                """,
                [table_name],
            ).fetchall()
        }
        if existing != set(columns):
            raise RuntimeError(
                f"Korean comment map does not match {table_name}: "
                f"missing={sorted(existing - set(columns))}, "
                f"extra={sorted(set(columns) - existing)}"
            )
        for column_name, comment in columns.items():
            escaped_comment = comment.replace("'", "''")
            connection.execute(
                f'COMMENT ON COLUMN "{table_name}"."{column_name}" ' f"IS '{escaped_comment}'"
            )

    rows = connection.execute(
        """
        SELECT table_name, column_name, column_comment
        FROM information_schema.columns
        WHERE table_name IN (
            'HF_TRNS_TRAN', 'CD_TRNS_TRAN', 'OB_INQR_TRAN',
            'OB_TRNS_TRAN', 'PI_WD_LEDG', 'GR_JC_TRAN',
            'GR_NJ_TRAN', 'CMS_REQ_TRAN', 'CMS_RES_TRAN'
        )
        """
    ).fetchall()
    actual = {(table, column): comment for table, column, comment in rows}
    expected = {
        (table, column): comment
        for table, columns in KOREAN_COLUMN_COMMENTS.items()
        for column, comment in columns.items()
    }
    if actual != expected:
        raise RuntimeError("Korean COMMENT verification failed")


def main() -> int:
    import argparse
    import duckdb

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("institution_a_synthetic_schema.duckdb"),
    )
    args = parser.parse_args()
    connection = duckdb.connect(str(args.database))
    try:
        apply_korean_comments(connection)
    finally:
        connection.close()
    print(f"Applied 92 Korean column comments to {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
