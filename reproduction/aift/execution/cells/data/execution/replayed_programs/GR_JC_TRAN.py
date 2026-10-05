def pseudonymize_data(df):
    df = df.copy()
    if 'CMS_NO' in df.columns:
        df['CMS_NO'] = df['CMS_NO'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    if 'DPSTR_NM' in df.columns:
        df['DPSTR_NM'] = df['DPSTR_NM'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    if 'WD_ACNO' in df.columns:
        df['WD_ACNO'] = df['WD_ACNO'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    return df
