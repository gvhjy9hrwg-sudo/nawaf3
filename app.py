import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
from datetime import datetime
import arabic_reshaper
from bidi.algorithm import get_display
import io

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def ar(text):
    return get_display(arabic_reshaper.reshape(str(text)))

def init_db():
    conn = sqlite3.connect("quality_reports.db")
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT, upload_date TEXT, rows_count INTEGER,
            cols_count INTEGER, null_count INTEGER, duplicate_count INTEGER,
            outlier_count INTEGER, format_errors INTEGER,
            quality_score REAL, rating TEXT
        )""")
    conn.commit(); conn.close()

def save_report(file_name, rows, cols, nulls, dups, outliers, fmt, score, rating):
    conn = sqlite3.connect("quality_reports.db")
    conn.cursor().execute("INSERT INTO reports VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
        (file_name, datetime.now().strftime("%Y-%m-%d %H:%M"), rows, cols, nulls, dups, outliers, fmt, round(score,2), rating))
    conn.commit(); conn.close()

def get_history():
    conn = sqlite3.connect("quality_reports.db")
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY id DESC", conn)
    conn.close(); return df

def analyze_nulls(df):
    c = df.isnull().sum()
    return c, (c/len(df))*100, c.sum()

def analyze_duplicates(df):
    return df.duplicated().sum(), df[df.duplicated(keep=False)]

def analyze_outliers(df):
    numeric = df.select_dtypes(include=[np.number]).columns
    total, details = 0, {}
    for col in numeric:
        Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3-Q1; lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
        mask = (df[col]<lo)|(df[col]>hi)
        rows = df[mask & df[col].notna()]
        if len(rows): details[col]={"count":len(rows),"lower":lo,"upper":hi,"rows":rows.index.tolist()}
        total += len(rows)
    return total, details, numeric

def analyze_format(df):
    err=0
    for col in df.select_dtypes(include=['object']).columns:
        err += df[col].dropna().apply(lambda x: isinstance(x,(int,float))).sum()
    return err

def calc_score(df, nulls, dups, outliers, fmt_err, numeric):
    cells = df.shape[0]*df.shape[1]
    s_null = ((cells-nulls)/cells)*100
    s_dup  = ((len(df)-dups)/len(df))*100
    tot_num = df[numeric].count().sum() if len(numeric)>0 else 1
    s_out  = ((tot_num-outliers)/tot_num)*100 if tot_num>0 else 100.0
    tot_obj = df.select_dtypes(include=['object']).size
    s_fmt  = ((tot_obj-fmt_err)/tot_obj)*100 if tot_obj>0 else 100.0
    final  = s_null*0.40 + s_dup*0.20 + s_out*0.20 + s_fmt*0.20
    return final, s_null, s_dup, s_out, s_fmt

def get_rating(score):
    if score>=90: return "ممتاز ✅","#00C853"
    elif score>=70: return "جيد 🟡","#FFD600"
    elif score>=50: return "متوسط 🟠","#FF6D00"
    else: return "ضعيف 🔴","#D50000"

def clean_data(df):
    df_clean = df.drop_duplicates()
    null_mask = df_clean.isnull().any(axis=1)
    clean_rows = df_clean[~null_mask].copy()
    dirty_rows = df_clean[null_mask].copy()
    if not dirty_rows.empty:
        notes = []
        for _, row in dirty_rows.iterrows():
            missing = [col for col in dirty_rows.columns if pd.isnull(row[col])]
            notes.append("ناقص: " + "، ".join(missing))
        dirty_rows["ملاحظة"] = notes
    result = pd.concat([clean_rows, dirty_rows], ignore_index=True)
    return result, len(df)-len(df_clean), clean_rows, dirty_rows

def color_clean(df):
    return pd.DataFrame('background-color: #1B5E20; color: white', index=df.index, columns=df.columns)

def color_dirty(df):
    return pd.DataFrame('background-color: #B71C1C; color: white', index=df.index, columns=df.columns)

init_db()

st.set_page_config(page_title="نظام تحليل جودة البيانات", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }
.hero { background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #01579b 100%); border-radius: 20px; padding: 50px 40px; text-align: center; margin-bottom: 30px; box-shadow: 0 20px 60px rgba(13,71,161,0.4); border: 1px solid rgba(255,255,255,0.1); }
.hero h1 { font-size: 2.8rem; font-weight: 800; color: white; margin: 0 0 10px 0; }
.hero p { font-size: 1.1rem; color: rgba(255,255,255,0.75); margin: 0; }
.metric-card { background: linear-gradient(145deg, #1a1f35, #0d1428); border-radius: 16px; padding: 24px 20px; text-align: center; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.metric-value { font-size: 2.2rem; font-weight: 800; color: #64B5F6; display: block; }
.metric-label { font-size: 0.9rem; color: rgba(255,255,255,0.55); margin-top: 6px; display: block; }
.score-ring { background: linear-gradient(135deg, #1a237e, #0d47a1); border-radius: 20px; padding: 35px; text-align: center; border: 2px solid rgba(100,181,246,0.3); }
.score-number { font-size: 4rem; font-weight: 800; color: white; line-height: 1; }
.score-label { font-size: 1.2rem; color: rgba(255,255,255,0.7); margin-top: 8px; }
.rec-card { background: linear-gradient(145deg, #1a2744, #0d1f3c); border-left: 4px solid #F39C12; border-radius: 12px; padding: 14px 18px; margin: 8px 0; color: #FDD835; font-weight: 500; }
.section-title { font-size: 1.4rem; font-weight: 700; color: white; margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid rgba(100,181,246,0.3); }
.progress-box { background: linear-gradient(145deg, #1B5E20, #2E7D32); border-radius: 16px; padding: 25px; text-align: center; border: 1px solid rgba(0,200,83,0.3); margin: 20px 0; }
.stTabs [data-baseweb="tab"] { background: #1a1f35; border-radius: 10px; color: rgba(255,255,255,0.6); border: 1px solid rgba(255,255,255,0.1); padding: 10px 24px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1565C0, #0d47a1) !important; color: white !important; }
div[data-testid="stFileUploader"] > label { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📊 نظام تحليل جودة البيانات الذكي</h1>
    <p>Smart Data Quality Validator — مشروع التخرج | قسم الحاسب الآلي وتقنية المعلومات</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔍  تحليل جديد", "🧹  تنظيف البيانات", "📁  سجل التقارير"])

with tabs[0]:
    st.markdown("### 📂 ارفع ملف البيانات (CSV أو Excel)")
    uploaded = st.file_uploader(" ", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        st.session_state['df'] = df
        st.session_state['fname'] = uploaded.name
        st.success(f"✅  تم رفع الملف بنجاح: **{uploaded.name}**")

        c1,c2,c3 = st.columns(3)
        for col, val, label in zip([c1,c2,c3], [df.shape[0], df.shape[1], df.shape[0]*df.shape[1]], ["عدد الصفوف","عدد الأعمدة","إجمالي الخلايا"]):
            with col:
                st.markdown(f'<div class="metric-card"><span class="metric-value">{val}</span><span class="metric-label">{label}</span></div>', unsafe_allow_html=True)

        with st.expander("👁️  معاينة البيانات — أول 5 صفوف"):
            st.dataframe(df.head(), use_container_width=True)

        null_counts, null_percent, total_nulls = analyze_nulls(df)
        dup_count, dup_df = analyze_duplicates(df)
        total_outliers, outlier_details, numeric_cols = analyze_outliers(df)
        format_errors = analyze_format(df)
        final_score, s_null, s_dup, s_out, s_fmt = calc_score(df, total_nulls, dup_count, total_outliers, format_errors, numeric_cols)
        rating, rating_color = get_rating(final_score)
        st.session_state['score'] = final_score

        st.markdown('<div class="section-title">📋 ملخص التحليل</div>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        for col, val, label, color in zip([m1,m2,m3,m4,m5],
            [f"{final_score:.1f}%", total_nulls, dup_count, total_outliers, format_errors],
            ["درجة الجودة","قيم ناقصة","صفوف مكررة","قيم شاذة","أخطاء تنسيق"],
            ["#64B5F6","#EF5350","#FFA726","#AB47BC","#26C6DA"]):
            with col:
                st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:{color}">{val}</span><span class="metric-label">{label}</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">📌 القيم الناقصة</div>', unsafe_allow_html=True)
            cols_with_nulls = null_counts[null_counts>0]
            if len(cols_with_nulls)==0:
                st.success("✅ لا توجد قيم ناقصة")
            else:
                for col in cols_with_nulls.index:
                    rows = [r+2 for r in df[df[col].isnull()].index.tolist()]
                    st.error(f"**{col}**: {null_counts[col]} قيمة ({null_percent[col]:.1f}%) — الصفوف: {rows}")

        with col_b:
            st.markdown('<div class="section-title">📌 الصفوف المكررة</div>', unsafe_allow_html=True)
            if dup_count==0:
                st.success("✅ لا توجد صفوف مكررة")
            else:
                st.warning(f"⚠️ عدد الصفوف المكررة: {dup_count}")
                id_col = df.columns[0]; name_col = df.columns[1] if len(df.columns)>1 else df.columns[0]
                for idx in df[df.duplicated()].index:
                    all_rows = [r+2 for r in df[df[id_col]==df.iloc[idx][id_col]].index.tolist()]
                    st.error(f"📍 الصف {idx+2}: **{df.iloc[idx][name_col]}** | {df.iloc[idx][id_col]} — يتكرر في الصفوف: {all_rows}")

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="section-title">📌 القيم الشاذة</div>', unsafe_allow_html=True)
            if total_outliers==0:
                st.success("✅ لا توجد قيم شاذة")
            else:
                for col, info in outlier_details.items():
                    rows = [r+2 for r in info["rows"]]
                    st.error(f"**{col}**: {info['count']} قيمة شاذة | النطاق: [{info['lower']:.1f} ← → {info['upper']:.1f}] | الصفوف: {rows}")

        with col_d:
            st.markdown('<div class="section-title">📌 أخطاء التنسيق</div>', unsafe_allow_html=True)
            if format_errors==0:
                st.success("✅ لا توجد أخطاء تنسيق")
            else:
                st.warning(f"⚠️ عدد أخطاء التنسيق: {format_errors}")

        st.markdown('<div class="section-title">⭐ درجة جودة البيانات</div>', unsafe_allow_html=True)
        col_score, col_recs = st.columns([1,2])

        with col_score:
            st.markdown(f"""<div class="score-ring">
                <div class="score-number" style="color:{rating_color}">{final_score:.1f}</div>
                <div class="score-label">/ 100 نقطة</div>
                <div style="color:{rating_color};font-size:1.3rem;font-weight:700;margin-top:12px">{rating}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(int(final_score)/100)

        with col_recs:
            st.markdown('<div class="section-title">💡 التوصيات</div>', unsafe_allow_html=True)
            if total_nulls>0:
                st.markdown(f'<div class="rec-card">🔧 عالج {total_nulls} قيمة ناقصة — راجع تبويب التنظيف</div>', unsafe_allow_html=True)
            if dup_count>0:
                st.markdown(f'<div class="rec-card">🔧 احذف {dup_count} صف مكرر — راجع تبويب التنظيف</div>', unsafe_allow_html=True)
            if total_outliers>0:
                st.markdown(f'<div class="rec-card">🔧 راجع {total_outliers} قيمة شاذة</div>', unsafe_allow_html=True)
            if format_errors>0:
                st.markdown(f'<div class="rec-card">🔧 صحّح {format_errors} خطأ تنسيق</div>', unsafe_allow_html=True)
            if total_nulls==0 and dup_count==0 and total_outliers==0 and format_errors==0:
                st.success("✅ بياناتك نظيفة تماماً")

        st.markdown('<div class="section-title">📈 التقرير البياني</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14,5), facecolor='#0d1428')
        fig.suptitle('Data Quality Report', fontsize=16, fontweight='bold', color='white')
        for ax in axes:
            ax.set_facecolor('#1a1f35'); ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax1 = axes[0]
        if null_counts.sum()>0:
            cols_plot = null_percent[null_percent>0]
            labels = [ar(c) for c in cols_plot.index]
            bars = ax1.bar(range(len(cols_plot)), cols_plot.values, color='#EF5350', edgecolor='#1a1f35', width=0.6)
            ax1.set_xticks(range(len(cols_plot))); ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color='white')
            for bar, val in zip(bars, cols_plot.values):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values!\nData is Clean', ha='center', va='center', fontsize=14, color='#00C853', transform=ax1.transAxes, fontweight='bold')
        ax1.set_title(ar('نسبة القيم الناقصة لكل عمود'), fontsize=13, color='white', pad=15)
        ax1.set_ylabel('Percentage (%)', color='white'); ax1.set_ylim(0,110); ax1.grid(axis='y', alpha=0.15, color='white')

        ax2 = axes[1]
        cats = [ar('القيم\nالناقصة'), ar('التكرارات'), ar('القيم\nالشاذة'), ar('التنسيق'), ar('الدرجة\nالنهائية')]
        vals = [s_null, s_dup, s_out, s_fmt, final_score]
        colors_bar = ['#EF5350','#FFA726','#AB47BC','#26C6DA','#64B5F6']
        bars2 = ax2.bar(cats, vals, color=colors_bar, edgecolor='#1a1f35', width=0.5)
        ax2.set_ylim(0,120); ax2.set_title(ar('تفصيل درجات الجودة'), fontsize=13, color='white', pad=15)
        ax2.set_ylabel('Score (%)', color='white')
        for bar, val in zip(bars2, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
        ax2.axhline(y=90, color='#00C853', linestyle='--', alpha=0.6, label='ممتاز (90%)')
        ax2.axhline(y=70, color='#FDD835', linestyle='--', alpha=0.6, label='جيد (70%)')
        ax2.axhline(y=50, color='#EF5350', linestyle='--', alpha=0.6, label='متوسط (50%)')
        ax2.legend(fontsize=8, labelcolor='white', facecolor='#1a1f35', edgecolor='#444')
        ax2.grid(axis='y', alpha=0.15, color='white')
        plt.tight_layout()
        st.pyplot(fig)

        save_report(uploaded.name, df.shape[0], df.shape[1], total_nulls, dup_count, total_outliers, format_errors, final_score, rating)
        st.info("💾 تم حفظ التقرير في قاعدة البيانات تلقائياً")

with tabs[1]:
    if 'df' not in st.session_state:
        st.info("📂 ارفع ملفاً أولاً من تبويب التحليل")
    else:
        df = st.session_state['df']
        old_score = st.session_state.get('score', 0)
        st.markdown('<div class="section-title">🧹 تنظيف البيانات وترتيبها</div>', unsafe_allow_html=True)
        st.markdown("""<div style="background:#1a1f35;border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.08);margin-bottom:20px">
            <p style="color:rgba(255,255,255,0.8);margin:0">
            ✅ حذف الصفوف المكررة<br>
            🟢 البيانات السليمة في الأعلى بخلفية خضراء<br>
            🔴 البيانات الناقصة في الأسفل بخلفية حمراء مع ملاحظة توضح وش الناقص<br>
            📥 تحميل الملف المنظم مباشرة من الشاشة
            </p></div>""", unsafe_allow_html=True)

        if st.button("🧹 نظّف البيانات الآن", use_container_width=True):
            cleaned, removed_dups, clean_rows, dirty_rows = clean_data(df)
            null_new = cleaned.isnull().sum().sum()
            dup_new = cleaned.duplicated().sum()
            numeric_new = cleaned.select_dtypes(include=[np.number]).columns
            out_new = 0
            for col in numeric_new:
                Q1,Q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
                IQR = Q3-Q1
                out_new += cleaned[(cleaned[col]<Q1-1.5*IQR)|(cleaned[col]>Q3+1.5*IQR)][col].count()
            fmt_new = analyze_format(cleaned)
            new_score,_,_,_,_ = calc_score(cleaned, null_new, dup_new, out_new, fmt_new, numeric_new)
            improvement = new_score - old_score

            st.markdown(f"""<div class="progress-box">
                <div style="color:white;font-size:1.1rem;margin-bottom:15px;font-weight:600">📊 نتيجة التنظيف</div>
                <div style="display:flex;justify-content:center;gap:40px;align-items:center">
                    <div style="text-align:center"><div style="font-size:2rem;font-weight:800;color:#EF5350">{old_score:.1f}%</div><div style="color:rgba(255,255,255,0.6)">قبل التنظيف</div></div>
                    <div style="font-size:2rem;color:white">←</div>
                    <div style="text-align:center"><div style="font-size:2rem;font-weight:800;color:#00C853">{new_score:.1f}%</div><div style="color:rgba(255,255,255,0.6)">بعد التنظيف</div></div>
                    <div style="text-align:center"><div style="font-size:1.8rem;font-weight:800;color:#64B5F6">+{improvement:.1f}%</div><div style="color:rgba(255,255,255,0.6)">التحسن</div></div>
                </div>
                <div style="color:rgba(255,255,255,0.7);margin-top:15px">تم حذف {removed_dups} صف مكرر · {len(clean_rows)} صف سليم · {len(dirty_rows)} صف ناقص</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title">📋 معاينة البيانات بعد التنظيف</div>', unsafe_allow_html=True)

            # دمج الجدولين في جدول واحد
            combined = pd.concat([clean_rows, dirty_rows], ignore_index=True)

            def color_combined(df):
                # كل الخلايا بيضاء افتراضياً
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for i, row in df.iterrows():
                    null_cols = [col for col in df.columns if pd.isnull(row[col])]
                    if null_cols:
                        # الصف ناقص — لوّن الخلايا الفارغة فقط باللون الأحمر
                        for col in null_cols:
                            styles.at[i, col] = 'background-color: #B71C1C; color: white'
                        # لوّن باقي الخلايا في الصف الناقص بأحمر فاتح
                        for col in df.columns:
                            if col not in null_cols and styles.at[i, col] == '':
                                styles.at[i, col] = 'background-color: #3a0a0a; color: white'
                    else:
                        # الصف سليم — أخضر
                        for col in df.columns:
                            styles.at[i, col] = 'background-color: #1B5E20; color: white'
                return styles

            st.dataframe(
                combined.style.apply(color_combined, axis=None),
                use_container_width=True
            )

            st.markdown('<div class="section-title">📥 تحميل الملف المنظم</div>', unsafe_allow_html=True)
            csv_buffer = io.StringIO()
            cleaned.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 تحميل الملف المنظم (CSV)",
                data=csv_buffer.getvalue().encode('utf-8-sig'),
                file_name=f"cleaned_{st.session_state.get('fname','data.csv')}",
                mime="text/csv",
                use_container_width=True
            )

with tabs[2]:
    st.markdown('<div class="section-title">📁 سجل التقارير السابقة</div>', unsafe_allow_html=True)
    history = get_history()
    if history.empty:
        st.info("لا توجد تقارير محفوظة بعد — حلّل ملفاً أولاً")
    else:
        st.dataframe(history, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#64B5F6">{len(history)}</span><span class="metric-label">إجمالي التقارير</span></div>', unsafe_allow_html=True)
        with c2:
            avg = history['quality_score'].mean()
            st.markdown(f'<div class="metric-card"><span class="metric-value" style="color:#00C853">{avg:.1f}%</span><span class="metric-label">متوسط درجة الجودة</span></div>', unsafe_allow_html=True)
