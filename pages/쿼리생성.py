import streamlit as st
import datetime
import re

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("📄 SQL 쿼리 생성기")

# --- 1. External Link ---
st.link_button("DD 플랫폼으로 이동 (쿼리 조회)", "https://nexon-sh.dd.deltaverse.cn/explore/index?space_id=2")

st.divider()

# --- 2. Query Generator ---
st.header("기간별 쿼리 생성")

# --- Date Calculation ---
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# --- Initialize Session State for Dates ---
if 'start_date' not in st.session_state:
    st.session_state.start_date = yesterday - datetime.timedelta(days=7)
if 'end_date' not in st.session_state:
    st.session_state.end_date = yesterday
if 'generated_query' not in st.session_state:
    st.session_state.generated_query = ""

# --- Query Template ---
query_template = """/*
거래소에서 거래된 장비들의 네트워크 분석을 위한 쿼리입니다.
*/-- 
set @start = '2025-10-22';
set @end = '2025-11-05';

with a as (
-- 거래타입 : 대금수령성공 일때의 로그
select dteventtime as 'sell_time', auction_no, izoneareaid, account_no as seller_account, char_no as seller_char, price, item_index, item_no, level as 'seller_lv'
from mg_dsl_log_auction_fht0
where 1=1
and log_auction_type = 8 --대금수령성공
),
b as (
-- 거래타입 : 구매성공 일때의 로그
select dteventtime as 'buy_time', auction_no, account_no as buyer_account, char_no as buyer_char, tier, gear_score, level as 'buyer_lv'
from mg_dsl_log_auction_fht0
where 1=1
and log_auction_type = 6
and dteventdate between @start and @end
),
reg_list as (
-- 거래소 등록 당시 로그에서 소울 등 아이템상세정보 추출
SELECT vopenid, vroleid, auction_no, item_index, item_no, regexp_extract(item_extra_option, '\\\\[1,\\\\\\d+,(\\\\\d+)', 1) as 'soul_index', item_extra_option, karma_scissors_count, upgrade_defaultvalue_level, item_level, transcendence_level, emblem_flyweight_index
FROM mg_dsl_log_auction_sales_registration_fht0 
where 1=1
and dteventdate between @start and @end

and item_index in (select DISTINCT item_index from mg_dsl_log_item_create_equip_fht0)
),
payment as (
    select vopenid, sum(pay_amt) / 100 as 'total_charge'
    from mg.ads_sr_mg_item_water_di
    group by vopenid
),
payment_a as (
    select a.*, payment.total_charge
    from a left join payment
    on a.seller_account = payment.vopenid
),
payment_b as (
    select b.*, payment.total_charge
    from b left join payment
    on b.buyer_account = payment.vopenid
),
item_names as (
select igoodsid, kor as item_name
from mg_nexon.meta_item_list
)

select 
a.izoneareaid,
a.sell_time as '판매시간',
a.seller_account as 'seller_vopenid', 
a.seller_char as 'seller_vroleid', 
a.seller_lv,
a.auction_no,
a.price as '거래가격', 
a.item_index, 
a.item_no,
coalesce(a.total_charge, 0) as 'seller 총과금액', 
b.buy_time as '구매시간',
b.buyer_account as 'buyer_vopenid', 
b.buyer_char as 'buyer_vroleid', 
b.buyer_lv,
b.tier, 
b.gear_score, 
coalesce(b.total_charge, 0) as 'buyer 총과금액', 
c.soul_index, 
c.item_extra_option, 
c.karma_scissors_count as '가위횟수',
c.upgrade_defaultvalue_level as '스타포스레벨',
c.item_level as '장비레벨',
c.transcendence_level as '초월레벨',
c.emblem_flyweight_index as '문장인덱스',
coalesce(d.item_name, a.item_index) as '아이템명',
coalesce(e.item_name, c.soul_index) as '소울'
from payment_a a join payment_b b
on a.auction_no = b.auction_no
join reg_list c
on a.auction_no = c.auction_no
left join item_names d
on a.item_index = d.igoodsid
left join item_names e
on c.soul_index = e.igoodsid;
"""

def generate_and_set_query():
    """세션 상태의 날짜를 기반으로 쿼리를 생성하고 세션 상태에 저장합니다."""
    start_date_str = st.session_state.start_date.strftime('%Y-%m-%d')
    end_date_str = st.session_state.end_date.strftime('%Y-%m-%d')

    query_with_start = re.sub(
        r"set @start = '.*';",
        f"set @start = '{start_date_str}';",
        query_template
    )
    final_query = re.sub(
        r"set @end = '.*';",
        f"set @end = '{end_date_str}';",
        query_with_start
    )
    st.session_state.generated_query = final_query

def set_date_range_and_generate(days):
    """프리셋에 대한 날짜를 설정하고 쿼리를 즉시 생성합니다."""
    st.session_state.end_date = yesterday
    st.session_state.start_date = yesterday - datetime.timedelta(days=days)
    generate_and_set_query()

# --- Preset Buttons ---
st.write("날짜 프리셋")
cols = st.columns(4)

with cols[0]:
    if st.button("최근 1주일", use_container_width=True):
        set_date_range_and_generate(7)
        st.rerun()

with cols[1]:
    if st.button("최근 1개월", use_container_width=True):
        set_date_range_and_generate(30)
        st.rerun()

with cols[2]:
    if st.button("최근 반년", use_container_width=True):
        set_date_range_and_generate(182)
        st.rerun()

with cols[3]:
    if st.button("최근 1년", use_container_width=True):
        set_date_range_and_generate(365)
        st.rerun()

# --- Date Input Widgets and Query Generation Button ---
st.write("---") # Add a separator for better visual grouping
col1, col2, col3 = st.columns(3)

with col1:
    start_date_input = st.date_input(
        "시작 날짜",
        key='start_date'
    )
with col2:
    end_date_input = st.date_input(
        "종료 날짜",
        key='end_date'
    )
with col3:
    # Add some vertical space to align the button with date inputs
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("쿼리 생성", use_container_width=True):
        generate_and_set_query()
        st.rerun()

if st.session_state.generated_query:
    st.subheader("생성된 SQL 쿼리")
    st.code(st.session_state.generated_query, language='sql')

