import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components
import json

# --- Constants & App Setup ---
FILTER_TYPE_ACCOUNT = "계정 (account)"
FILTER_TYPE_CHAR = "캐릭터 (char)"
NODE_SIZE_거래가격 = "거래가격"
NODE_SIZE_CONNECTION = "connection"
MAX_NODES_TO_RENDER = 700

def initialize_session_state():
    """세션 상태의 기본값을 설정합니다."""
    defaults = {
        'base_edge_data': None,
        'df_filtered_original': None,
        'base_detail_data': None,
        'all_node_ids': [],
        'force_render': False,
        'amount_threshold': 0,
        'node_size_standard': NODE_SIZE_거래가격,
        'min_거래가격': 0,
        'filter_type': FILTER_TYPE_ACCOUNT,
        'filter_values_text': "",
        'filter_logic_type': "관계 기준",
        'item_no_filter_text': "",
        'top_n_filter_type': "없음",
        'top_n_value': 0,
        'min_mutual_transaction_count': 0,
        'custom_graph_title': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ----------------------------------------------------------------------
# 데이터 처리 및 그래프 생성 함수
# ----------------------------------------------------------------------

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
        return None
    df['seller_vopenid'] = df['seller_vopenid'].astype(str)
    df['buyer_vopenid'] = df['buyer_vopenid'].astype(str)
    df['seller_vroleid'] = df['seller_vroleid'].astype(str)
    df['buyer_vroleid'] = df['buyer_vroleid'].astype(str)
    df['item_no'] = df['item_no'].astype(str)
    df['거래가격'] = pd.to_numeric(df['거래가격'], errors='coerce')
    df = df.dropna(subset=['거래가격', 'seller_vopenid', 'buyer_vopenid'])
    return df

def search_df(data, account_no):
    query = str(account_no)
    return data[(data['seller_vopenid'].astype(str).str.contains(query)) | 
                (data['buyer_vopenid'].astype(str).str.contains(query))]

@st.cache_data
def data_processing_by_관계거래가격(df, amount):
    df_edge = df.groupby(['seller_vopenid', 'buyer_vopenid']).agg(
        transaction_count=('auction_no', 'count'),
        total_거래가격=('거래가격', 'sum')
    ).reset_index()
    a = df_edge[df_edge['total_거래가격'] > amount]
    if a.empty:
        return pd.DataFrame(columns=df_edge.columns), pd.DataFrame(columns=df.columns)
    c = pd.unique(a[['seller_vopenid', 'buyer_vopenid']].values.ravel('K'))
    data_filtered = df[df['seller_vopenid'].isin(c) | df['buyer_vopenid'].isin(c)]
    edge_data = data_filtered.groupby(['seller_vopenid', 'buyer_vopenid']).agg(
        transaction_count=('auction_no', 'count'),
        total_거래가격=('거래가격', 'sum')
    ).reset_index()
    return edge_data, data_filtered

@st.cache_data
def data_processing_by_계정거래가격(df, amount):
    if df.empty:
        return pd.DataFrame(columns=['seller_vopenid', 'buyer_vopenid', 'transaction_count', 'total_거래가격']), pd.DataFrame(columns=df.columns)
    seller_totals = df.groupby('seller_vopenid')['거래가격'].sum()
    buyer_totals = df.groupby('buyer_vopenid')['거래가격'].sum()
    all_accounts = pd.concat([seller_totals, buyer_totals]).groupby(level=0).sum()
    filtered_accounts = all_accounts[all_accounts > amount].index.tolist()
    if not filtered_accounts:
        return pd.DataFrame(columns=['seller_vopenid', 'buyer_vopenid', 'transaction_count', 'total_거래가격']), pd.DataFrame(columns=df.columns)
    data_filtered = df[df['seller_vopenid'].isin(filtered_accounts) | df['buyer_vopenid'].isin(filtered_accounts)]
    edge_data = data_filtered.groupby(['seller_vopenid', 'buyer_vopenid']).agg(
        transaction_count=('auction_no', 'count'),
        total_거래가격=('거래가격', 'sum')
    ).reset_index()
    return edge_data, data_filtered

def network_graph(edge_data, original_df, title_text, standard=NODE_SIZE_CONNECTION):
    G = nx.DiGraph()
    for _, row in edge_data.iterrows():
        G.add_edge(row['seller_vopenid'], row['buyer_vopenid'], weight=row['transaction_count'], 거래가격=row['total_거래가격'])
    if not G.nodes():
        return go.Figure(layout=go.Layout(title="표시할 데이터가 없습니다.")), json.dumps([])
    pos = nx.spring_layout(G, seed=42)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]
    buyer_amounts = original_df.groupby('buyer_vopenid')['거래가격'].sum().to_dict()
    seller_amounts = original_df.groupby('seller_vopenid')['거래가격'].sum().to_dict()
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    middle_node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', marker=dict(opacity=0))
    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        middle_node_trace['x'] += tuple([(x0 + x1) / 2])
        middle_node_trace['y'] += tuple([(y0 + y1) / 2])
        middle_node_trace['text'] += tuple([f"거래 횟수: {edge[2]['weight']}<br>총 거래액: {edge[2]['거래가격']:,.0f}"])
    node_x, node_y, node_text, node_colors, node_sizes, node_ids = [], [], [], [], [], []
    edge_거래가격s = [s[-1]['거래가격'] for s in G.edges(data=True)]
    devider = np.mean(edge_거래가격s) if edge_거래가격s and np.mean(edge_거래가격s) != 0 else 1
    active_sellers = set(edge_data['seller_vopenid'].values)
    active_buyers = set(edge_data['buyer_vopenid'].values)
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))
        connections = G.degree(node, weight='weight')
        거래가격_weight = G.degree(node, weight='거래가격')
        if standard == NODE_SIZE_CONNECTION:
            node_sizes.append(10 + (np.log10(max(1, connections)) * 10))
        elif standard == NODE_SIZE_거래가격:
            node_sizes.append(10 + (거래가격_weight / devider))
        is_seller, is_buyer = node in active_sellers, node in active_buyers
        seller_거래가격, buyer_거래가격 = seller_amounts.get(node, 0), buyer_amounts.get(node, 0)
        if is_seller and is_buyer:
            node_type, color = "Seller & Buyer", "purple"
            text = f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>Seller Total 거래가격: {seller_거래가격:,.0f}<br>Buyer Total 거래가격: {buyer_거래가격:,.0f}"
        elif is_seller:
            node_type, color = "Seller", "blue"
            text = f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>Seller Total 거래가격: {seller_거래가격:,.0f}"
        else:
            node_type, color = "Buyer", "green"
            text = f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>Buyer Total 거래가격: {buyer_거래가격:,.0f}"
        node_colors.append(color)
        node_text.append(text)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, customdata=node_ids, marker=dict(color=node_colors, size=node_sizes, line_width=2))
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    adj_list = [[node_map[neighbor] for neighbor in set(list(G.successors(node)) + list(G.predecessors(node)))] for node in node_list]
    fig = go.Figure(data=[edge_trace, node_trace, middle_node_trace], layout=go.Layout(title=dict(text=title_text, font=dict(size=16)), showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig, json.dumps(adj_list)

def reset_all_settings():
    keys_to_delete = ['base_edge_data', 'df_filtered_original', 'base_detail_data', 'all_node_ids', 'force_render', 'amount_threshold', 'node_size_standard', 'min_거래가격', 'filter_type', 'filter_values_text', 'filter_logic_type', 'item_no_filter_text', 'top_n_filter_type', 'top_n_value', 'min_mutual_transaction_count', 'custom_graph_title', 'filter_file', 'item_no_filter_file']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

def display_graph(node_count, selected_account):
    st.subheader("📈 네트워크 그래프")
    if node_count > MAX_NODES_TO_RENDER and not st.session_state.get('force_render', False):
        st.error(f"⚠️ **성능 경고:** 시각화할 노드의 개수({node_count}개)가 너무 많습니다.")
        if st.button("그래도 그래프 생성하기 (앱이 멈출 수 있습니다)"):
            st.session_state.force_render = True
            st.rerun()
        st.warning(f"느린 속도를 원치 않으시면, 사이드바의 '기준 총 거래액'을 높여 노드 개수를 {MAX_NODES_TO_RENDER}개 이하로 줄여주세요.")
        return
    display_edge_data = search_df(st.session_state.base_edge_data, selected_account) if selected_account != "-- 전체 보기 --" else st.session_state.base_edge_data
    custom_title = st.session_state.get('custom_graph_title', '').strip()
    if custom_title:
        title_text = custom_title
    elif selected_account == "-- 전체 보기 --":
        title_text = f"전체 거래 네트워크 (기준금액: {st.session_state.amount_threshold:,.0f})"
    else:
        title_text = f"'{selected_account}' 계정 거래 네트워크"
    if display_edge_data.empty:
        st.warning("선택한 조건에 맞는 그래프 데이터가 없습니다.")
    else:
        fig, adj_list_json = network_graph(display_edge_data, st.session_state.df_filtered_original, title_text=title_text, standard=st.session_state.node_size_standard)
        graph_json = fig.to_json()
        js_script = f'''<script src="https://cdn.plot.ly/plotly-latest.min.js"></script><div id="plotly-graph-div"></div><script>var spec = {graph_json};var adj = {adj_list_json};var graphDiv = document.getElementById('plotly-graph-div');Plotly.newPlot(graphDiv, spec.data, spec.layout);function fallbackCopyToClipboard(text){{var textArea=document.createElement("textarea");textArea.value=text;textArea.style.top="0";textArea.style.left="0";textArea.style.position="fixed";document.body.appendChild(textArea);textArea.focus();textArea.select();try{{var successful=document.execCommand('copy');if(successful)alert('계정 ID가 클립보드에 복사되었습니다: '+text);else alert('클립보드 복사에 실패했습니다.');}}catch(err){{console.error('Fallback clipboard copy failed:',err);alert('클립보드 복사에 실패했습니다.');}}document.body.removeChild(textArea);}}function copyToClipboard(text){{if(navigator.clipboard&&window.isSecureContext){{navigator.clipboard.writeText(text).then(function(){{alert('계정 ID가 클립보드에 복사되었습니다: '+text);}},function(err){{fallbackCopyToClipboard(text);}});}}else{{fallbackCopyToClipboard(text);}}}}graphDiv.on('plotly_click',function(data){{if(data.points.length>0){{var point=data.points[0];if(point.curveNumber===1&&point.customdata){{copyToClipboard(point.customdata);}}}}}});graphDiv.on('plotly_hover',function(data){{if(data.points.length>0){{var point=data.points[0];if(point.curveNumber===1){{var pointNumber=point.pointNumber;var neighbors=adj[pointNumber];var numNodes=spec.data[1].x.length;var opacities=Array(numNodes).fill(0.2);opacities[pointNumber]=1.0;neighbors.forEach(function(neighborIdx){{opacities[neighborIdx]=1.0;}});Plotly.restyle(graphDiv,{{'marker.opacity':[opacities]}},[1]);}}}}}});graphDiv.on('plotly_unhover',function(data){{Plotly.restyle(graphDiv,{{'marker.opacity':1}},[1]);}});</script>'''
        components.html(js_script, height=800, scrolling=False)

def display_table(selected_account):
    st.subheader("📊 상세 거래 데이터")
    display_detail_data = st.session_state.base_detail_data if selected_account == "-- 전체 보기 --" else st.session_state.base_detail_data[(st.session_state.base_detail_data['seller_vopenid'] == selected_account) | (st.session_state.base_detail_data['buyer_vopenid'] == selected_account)]
    all_possible_cols = ['izoneareaid', '판매시간', 'seller_vopenid', 'seller_vroleid', 'seller_lv', 'auction_no', '거래가격', 'item_index', 'item_no', 'seller 총과금액', '구매시간', 'buyer_vopenid', 'buyer_vroleid', 'buyer_lv', 'tier', 'gear_score', 'buyer 총과금액', 'soul_index', 'item_extra_option', '가위횟수', '스타포스레벨', '장비레벨', '초월레벨', '문장인덱스', '아이템명', '소울']
    default_cols = ['판매시간', 'seller_vopenid', 'buyer_vopenid', '거래가격', 'gear_score', '아이템명', '가위횟수', '스타포스레벨', '장비레벨', '초월레벨', '문장인덱스', '소울']
    if not display_detail_data.empty:
        available_cols = [col for col in all_possible_cols if col in display_detail_data.columns]
        default_cols = [col for col in default_cols if col in available_cols]
        if 'column_selection_state' not in st.session_state:
            st.session_state.column_selection_state = default_cols
        st.write("##### 컬럼 프리셋 선택")
        cols = st.columns([4, 1, 5, 5, 7, 5])
        if cols[0].button("기본 컬럼으로 복원", use_container_width=True):
            st.session_state.column_selection_state = default_cols
            st.rerun()
        cols[1].markdown('<div style="height: 28px; display: flex; align-items: center; justify-content: center;">|</div>', unsafe_allow_html=True)
        preset_cols = {"판매자 정보": ['판매시간', 'seller_vopenid', 'seller_vroleid', 'seller_lv', 'seller 총과금액'], "구매자 정보": ['구매시간', 'buyer_vopenid', 'buyer_vroleid', 'buyer_lv', 'buyer 총과금액'], "거래아이템 상세 정보": ['item_no', 'item_index', 'tier', '거래가격', 'gear_score', '가위횟수', '스타포스레벨', '장비레벨', '초월레벨', '문장인덱스', '아이템명', '소울']}
        if cols[2].button("판매자 정보", use_container_width=True):
            st.session_state.column_selection_state += [col for col in preset_cols["판매자 정보"] if col in available_cols and col not in st.session_state.column_selection_state]
            st.rerun()
        if cols[3].button("구매자 정보", use_container_width=True):
            st.session_state.column_selection_state += [col for col in preset_cols["구매자 정보"] if col in available_cols and col not in st.session_state.column_selection_state]
            st.rerun()
        if cols[4].button("거래아이템 상세 정보", use_container_width=True):
            st.session_state.column_selection_state += [col for col in preset_cols["거래아이템 상세 정보"] if col in available_cols and col not in st.session_state.column_selection_state]
            st.rerun()
        if cols[5].button("전부 비우기", use_container_width=True):
            st.session_state.column_selection_state = []
            st.rerun()
        selected_cols = st.multiselect("표시할 컬럼을 최종 선택하세요.", available_cols, default=st.session_state.column_selection_state, label_visibility="collapsed")
        st.session_state.column_selection_state = selected_cols
        if selected_cols:
            df_to_show = display_detail_data[selected_cols].copy()
            if '거래가격' in df_to_show.columns:
                df_to_show.sort_values(by="거래가격", ascending=False, inplace=True)
            st.dataframe(df_to_show)
            csv_data = df_to_show.to_csv(index=False).encode('utf-8-sig')
            file_name = f'detail_{selected_account}.csv' if selected_account != "-- 전체 보기 --" else "detail_all.csv"
            st.download_button("📥 CSV로 다운로드", csv_data, file_name, 'text/csv')
        else:
            st.warning("표시할 컬럼을 하나 이상 선택해주세요.")
    else:
        st.info("표시할 상세 거래 데이터가 없습니다.")

def display_main_content():
    if 'base_edge_data' not in st.session_state or st.session_state.base_edge_data is None:
        st.info("사이드바에서 옵션을 설정한 후 '그래프 생성' 버튼을 눌러주세요.")
        return
    if not st.session_state.base_edge_data.empty:
        node_count = pd.unique(st.session_state.base_edge_data[['seller_vopenid', 'buyer_vopenid']].values.ravel('K')).size
    else:
        node_count = 0
    st.subheader("🔍 계정 ID로 필터링")
    st.caption("그래프와 하단 테이블에 모두 적용됩니다.")
    filter_options = ["-- 전체 보기 --"] + st.session_state.all_node_ids
    selected_account = st.selectbox("필터링할 계정 ID를 선택하세요:", options=filter_options, index=0, label_visibility="collapsed")
    display_graph(node_count, selected_account)
    display_table(selected_account)

# ----------------------------------------------------------------------
# 메인 앱 실행 로직
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
initialize_session_state()

st.title("📈 네트워크 분석")

uploaded_file = st.file_uploader(
    "거래 내역 CSV 또는 Excel 파일을 업로드하세요.", 
    type=["csv", "xlsx"],
    help="이 페이지의 모든 기능은 파일이 업로드되어야 활성화됩니다."
)

if uploaded_file is not None:
    df_original = load_data(uploaded_file)
    
    with st.sidebar:
        st.header("⚙️ 그래프 생성 옵션")
        with st.form(key='settings_form'):
            with st.expander("1. 그래프 구성", expanded=True):
                st.selectbox("금액 필터링 기준", ["관계 기준", "계정 기준"], key='filter_logic_type', help="- 관계 기준: 판매자-구매자 관계의 '총 거래액'을 기준으로 필터링합니다.\n- 계정 기준: 각 계정의 '총 거래액(판매+구매)'을 기준으로 필터링합니다.")
                st.number_input("기준 총 거래액", min_value=0, key='amount_threshold', help="선택한 필터링 기준에 따라 이 금액을 초과하는 대상을 필터링합니다.")
                st.number_input("최소 상호 거래 횟수", min_value=0, key='min_mutual_transaction_count', help="두 계정 간의 최소 거래 횟수를 설정합니다. 이 횟수 미만의 연결은 그래프에서 제외됩니다.")
                st.selectbox("노드(원) 크기 기준", [NODE_SIZE_거래가격, NODE_SIZE_CONNECTION], key='node_size_standard')
                st.text_input("그래프 제목 (선택 사항)", key='custom_graph_title', placeholder="입력 시 기본 제목을 덮어씁니다.")
            
            with st.expander("고급 필터링"):
                st.subheader("데이터 필터링")
                st.radio("특정 계정/캐릭터 필터", [FILTER_TYPE_ACCOUNT, FILTER_TYPE_CHAR], key='filter_type', help="특정 계정 또는 캐릭터와 관련된 거래만 필터링합니다.")
                st.text_area("Vopenid 또는 Vroleid 목록 입력", placeholder="쉼표(,)로 구분하여 여러 개 입력", key='filter_values_text')
                st.file_uploader("또는 CSV/Excel 파일 업로드 (첫 번째 열 사용)", type=['csv', 'xlsx'], key='filter_file')
                st.number_input("최소 개별 거래액", min_value=0, key='min_거래가격', help="이 금액 미만인 개별 거래는 최초 데이터에서 제외합니다.")
                st.divider()
                st.subheader("아이템 필터링")
                st.text_area("Item No 목록 입력", placeholder="쉼표(,)로 구분하여 여러 개 입력", key='item_no_filter_text')
                st.file_uploader("또는 Item No 목록 파일 업로드 (첫 번째 열 사용)", type=['csv', 'xlsx'], key='item_no_filter_file')
                st.divider()
                st.subheader("Top N 필터링")
                st.selectbox("Top N 기준", ["없음", "거래금액 상위", "총 거래횟수 상위"], key='top_n_filter_type', help="- 거래금액 상위: 총 거래액(판매+구매)이 가장 높은 N개의 계정을 필터링합니다.\n- 총 거래횟수 상위: 총 거래 횟수(판매+구매)가 가장 많은 N개의 계정을 필터링합니다.")
                st.number_input("상위 N명", min_value=0, key='top_n_value', help="필터링할 상위 계정의 수를 입력하세요. 0이면 적용되지 않습니다.")

            submitted = st.form_submit_button("🚀 그래프 생성")

        sample_df = pd.DataFrame({'ID': ["sample_id_1", "sample_id_2"]})
        sample_csv = sample_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 샘플 CSV 다운로드", data=sample_csv, file_name="filter_sample.csv", mime="text/csv")
        
        if st.button("🔄 설정 초기화"):
            reset_all_settings()
            st.rerun()

    if 'submitted' not in locals():
        submitted = False

    if submitted:
        with st.spinner("데이터 처리 중..."):
            st.session_state.force_render = False
            df_to_process = df_original.copy()
            filter_list = []
            if st.session_state.filter_values_text:
                filter_list.extend([v.strip() for v in st.session_state.filter_values_text.split(',') if v.strip()])
            filter_file_data = st.session_state.get('filter_file')
            if filter_file_data:
                try:
                    df_filter = pd.read_excel(filter_file_data) if filter_file_data.name.endswith('.xlsx') else pd.read_csv(filter_file_data)
                    if not df_filter.empty:
                        filter_list.extend(df_filter.iloc[:, 0].astype(str).tolist())
                except Exception as e:
                    st.sidebar.error(f"필터 파일 처리 중 오류: {e}")
            filter_list = list(set(filter_list))
            if filter_list:
                query_regex = '|'.join(filter_list)
                if st.session_state.filter_type == FILTER_TYPE_ACCOUNT:
                    df_to_process = df_to_process[df_to_process['seller_vopenid'].str.contains(query_regex, na=False) | df_to_process['buyer_vopenid'].str.contains(query_regex, na=False)]
                elif st.session_state.filter_type == FILTER_TYPE_CHAR:
                    df_to_process = df_to_process[df_to_process['seller_vroleid'].str.contains(query_regex, na=False) | df_to_process['buyer_vroleid'].str.contains(query_regex, na=False)]
            item_filter_list = []
            if st.session_state.item_no_filter_text:
                item_filter_list.extend([v.strip() for v in st.session_state.item_no_filter_text.split(',') if v.strip()])
            item_filter_file_data = st.session_state.get('item_no_filter_file')
            if item_filter_file_data:
                try:
                    df_item_filter = pd.read_excel(item_filter_file_data) if item_filter_file_data.name.endswith('.xlsx') else pd.read_csv(item_filter_file_data)
                    if not df_item_filter.empty:
                        item_filter_list.extend(df_item_filter.iloc[:, 0].astype(str).tolist())
                except Exception as e:
                    st.sidebar.error(f"아이템 필터 파일 처리 중 오류: {e}")
            item_filter_list = list(set(item_filter_list))
            if item_filter_list:
                df_to_process = df_to_process[df_to_process['item_no'].isin(item_filter_list)]
            df_filtered = df_to_process[df_to_process['거래가격'] >= st.session_state.min_거래가격].copy()
            if st.session_state.filter_logic_type == "계정 기준":
                base_data, base_details = data_processing_by_계정거래가격(df_filtered, amount=st.session_state.amount_threshold)
            else:
                base_data, base_details = data_processing_by_관계거래가격(df_filtered, amount=st.session_state.amount_threshold)
            min_count = st.session_state.min_mutual_transaction_count
            if min_count > 1 and not base_data.empty:
                base_data = base_data[base_data['transaction_count'] >= min_count].copy()
                if not base_data.empty:
                    filtered_nodes = pd.unique(base_data[['seller_vopenid', 'buyer_vopenid']].values.ravel('K'))
                    base_details = base_details[base_details['seller_vopenid'].isin(filtered_nodes) | base_details['buyer_vopenid'].isin(filtered_nodes)].copy()
                else:
                    base_details = pd.DataFrame(columns=base_details.columns)
            top_n_type = st.session_state.top_n_filter_type
            top_n_value = st.session_state.top_n_value
            if top_n_type != "없음" and top_n_value > 0 and not base_data.empty:
                top_n_nodes = []
                if top_n_type == "거래금액 상위":
                    seller_totals = base_details.groupby('seller_vopenid')['거래가격'].sum()
                    buyer_totals = base_details.groupby('buyer_vopenid')['거래가격'].sum()
                    all_accounts = pd.concat([seller_totals, buyer_totals]).groupby(level=0).sum()
                    top_n_nodes = all_accounts.nlargest(top_n_value).index.tolist()
                elif top_n_type == "총 거래횟수 상위":
                    seller_counts = base_data.groupby('seller_vopenid')['transaction_count'].sum()
                    buyer_counts = base_data.groupby('buyer_vopenid')['transaction_count'].sum()
                    all_counts = pd.concat([seller_counts, buyer_counts]).groupby(level=0).sum()
                    if not all_counts.empty:
                        top_n_nodes = all_counts.nlargest(top_n_value).index.tolist()
                if top_n_nodes:
                    base_data = base_data[base_data['seller_vopenid'].isin(top_n_nodes) | base_data['buyer_vopenid'].isin(top_n_nodes)]
                    base_details = base_details[base_details['seller_vopenid'].isin(top_n_nodes) | base_details['buyer_vopenid'].isin(top_n_nodes)]
            st.session_state.base_edge_data = base_data
            st.session_state.df_filtered_original = df_filtered
            st.session_state.base_detail_data = base_details
            if not base_data.empty:
                st.session_state.all_node_ids = sorted(list(pd.unique(base_data[['seller_vopenid', 'buyer_vopenid']].values.ravel('K'))))
            else:
                st.session_state.all_node_ids = []
    
    display_main_content()
else:
    st.info("분석을 시작하려면 거래 내역 파일을 업로드해주세요. 파일이 없는 경우 쿼리 생성 페이지를 방문하세요.")
