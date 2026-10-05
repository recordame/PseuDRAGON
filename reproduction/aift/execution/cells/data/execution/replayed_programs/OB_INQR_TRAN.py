def pseudonymize_data(df):
    df = df.copy()
    if 'FIN_INTT_SVC_NO' in df.columns:
        df['FIN_INTT_SVC_NO'] = df['FIN_INTT_SVC_NO'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    if 'USER_SEQ_NO' in df.columns:
        df['USER_SEQ_NO'] = df['USER_SEQ_NO'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: f'{int(x) // 1000 * 1000}~{int(x) // 1000 * 1000 + 999}' if pd.notna(x) and str(x).isdigit() else str(x))(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    if 'CI' in df.columns:
        df['CI'] = df['CI'].apply(lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: (lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None)(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value)
    return df
