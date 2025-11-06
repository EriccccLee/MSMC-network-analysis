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
NODE_SIZE_PRICE = "price"
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
        'node_size_standard': NODE_SIZE_PRICE,
        'min_price': 0,
        'filter_type': FILTER_TYPE_ACCOUNT,
        'filter_value': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ----------------------------------------------------------------------
# 원본 코드 함수 (일부 수정됨)
# ----------------------------------------------------------------------

# 1. search_df 함수 (원본 유지)
def search_df(data, account_no):
    """(필터링용) 집계된 엣지 데이터에서 특정 계정 번호로 필터링합니다."""
    query = str(account_no)
    return data[(data['seller_account'].astype(str).str.contains(query)) | 
                (data['buyer_account'].astype(str).str.contains(query))]

# 2. data_processing_by_price 함수 (성능 개선)
@st.cache_data
def data_processing_by_price(df, amount):
    """
    거래 금액 기준으로 데이터를 집계하고, 상세 데이터도 함께 반환합니다.
    (캐싱 적용)
    """
    df_edge = df.groupby(['seller_account', 'buyer_account']).agg(
        transaction_count=('auction_no', 'count'),
        total_price=('price', 'sum')
    ).reset_index()
    
    a = df_edge[df_edge['total_price'] > amount]

    # 기준을 충족하는 거래가 없는 경우 빈 데이터프레임 반환
    if a.empty:
        empty_edges = pd.DataFrame(columns=['seller_account', 'buyer_account', 'transaction_count', 'total_price'])
        empty_details = pd.DataFrame(columns=df.columns)
        return empty_edges, empty_details

    b = pd.concat([a['seller_account'], a['buyer_account']])
    c = list(set(b))
    
    data_filtered = df[df['seller_account'].isin(c) | df['buyer_account'].isin(c)]
    
    edge_data = data_filtered.groupby(['seller_account', 'buyer_account']).agg(
        transaction_count=('auction_no', 'count'),
        total_price=('price', 'sum')
    ).reset_index()
    
    return edge_data, data_filtered # 그래프용 집계 데이터와 테이블용 상세 데이터 모두 반환

# 3. network_graph 함수 (원본과 거의 동일)
def network_graph(edge_data, original_df, title_text, standard=NODE_SIZE_CONNECTION):
    """
    집계된 엣지 데이터(edge_data)와 원본 데이터(original_df)를 기반으로
    Plotly 네트워크 그래프를 생성하고 Figure 객체와 하이라이팅을 위한 인접 리스트를 반환합니다.
    """
    G = nx.DiGraph()

    for _, row in edge_data.iterrows():
        G.add_edge(row['seller_account'], row['buyer_account'], weight=row['transaction_count'], price=row['total_price'])
        
    if not G.nodes():
        return go.Figure(layout=go.Layout(title="표시할 데이터가 없습니다.")), json.dumps([])

    pos = nx.spring_layout(G, seed=42)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    buyer_amounts = original_df.groupby('buyer_account')['price'].sum().to_dict()
    seller_amounts = original_df.groupby('seller_account')['price'].sum().to_dict()

    # 1. Edge Trace (Lines)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 2. Edge Hover Trace (Invisible markers at midpoint)
    middle_node_trace = go.Scatter(
        x=[], y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(opacity=0)
    )
    for edge in G.edges(data=True):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        middle_node_trace['x'] += tuple([(x0 + x1) / 2])
        middle_node_trace['y'] += tuple([(y0 + y1) / 2])
        weight = edge[2]['weight']
        price = edge[2]['price']
        middle_node_trace['text'] += tuple([f"거래 횟수: {weight}<br>총 거래액: {price:,.0f}"])

    # 3. Node Trace
    node_x, node_y, node_adjacencies, node_text, node_colors, node_sizes, node_ids = [], [], [], [], [], [], []
    
    edge_prices = [s[-1]['price'] for s in G.edges(data=True)]
    if edge_prices:
        devider = np.mean(edge_prices)
        if devider == 0: devider = 1
    else:
        devider = 1 

    active_sellers = set(edge_data['seller_account'].values)
    active_buyers = set(edge_data['buyer_account'].values)

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node)) # 클립보드 복사를 위해 str로 변환하여 ID 저장
        
        connections = G.degree(node, weight='weight')
        price_weight = G.degree(node, weight='price')
        node_adjacencies.append(connections)
        
        if standard == NODE_SIZE_CONNECTION:
            node_sizes.append(10 + (connections * 2))
        elif standard == NODE_SIZE_PRICE:
            node_sizes.append(10 + (price_weight / devider))

        is_seller = node in active_sellers
        is_buyer = node in active_buyers
        
        seller_price = seller_amounts.get(node, 0)
        buyer_price = buyer_amounts.get(node, 0)

        if is_seller and is_buyer:
            node_type = "Seller & Buyer"
            node_colors.append('purple')
            node_text.append(
                f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>"
                f"Seller Total Price: {seller_price:,.0f}<br>Buyer Total Price: {buyer_price:,.0f}"
            )
        elif is_seller:
            node_type = "Seller"
            node_colors.append('blue')
            node_text.append(
                f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>"
                f"Seller Total Price: {seller_price:,.0f}"
            )
        else:
            node_type = "Buyer"
            node_colors.append('green')
            node_text.append(
                f"Account Type: {node_type}<br>Account ID: {node}<br># of connections: {connections}<br>"
                f"Buyer Total Price: {buyer_price:,.0f}"
            )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        customdata=node_ids, # 클릭 시 ID를 가져오기 위해 customdata 설정
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=2))

    # 4. Adjacency list for highlighting
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    adj_list = []
    for node in node_list:
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        adj_list.append([node_map[neighbor] for neighbor in set(neighbors)])

    fig = go.Figure(data=[edge_trace, node_trace, middle_node_trace],
                 layout=go.Layout(
                     title=dict(text=title_text, font=dict(size=16)),
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig, json.dumps(adj_list)

# ----------------------------------------------------------------------
# Streamlit 앱 구현
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("[MSMC] 장비 거래 네트워크 분석앱")


# 1. 파일 업로드
uploaded_file = st.file_uploader("거래 내역 CSV 파일을 업로드하세요.", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['seller_account'] = df['seller_account'].astype(str)
    df['buyer_account'] = df['buyer_account'].astype(str)
    df['seller_char'] = df['seller_char'].astype(str)
    df['buyer_char'] = df['buyer_char'].astype(str)
    df['item_no'] = df['item_no'].astype(str)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price', 'seller_account', 'buyer_account'])
    return df

if uploaded_file is not None:
    df_original = load_data(uploaded_file)
    
    # --- 사이드바 ---
    st.sidebar.header("⚙️ 그래프 생성 옵션")

    # 콜백 함수를 먼저 정의합니다.
    def generate_graph_data():
        """
        버튼 클릭 시 호출될 콜백 함수.
        데이터를 처리하여 session_state에 저장합니다.
        """
        with st.spinner("데이터 처리 중..."):
            st.session_state.force_render = False # 강제 렌더링 상태 초기화
            
            df_to_process = df_original.copy()

            # session_state에서 필터 값 가져오기
            if 'filter_value' in st.session_state and st.session_state.filter_value:
                query = str(st.session_state.filter_value)
                if st.session_state.filter_type == FILTER_TYPE_ACCOUNT:
                    df_to_process = df_to_process[
                        (df_to_process['seller_account'].astype(str).str.contains(query)) |
                        (df_to_process['buyer_account'].astype(str).str.contains(query))
                    ]
                elif st.session_state.filter_type == FILTER_TYPE_CHAR:
                    df_to_process = df_to_process[
                        (df_to_process['seller_char'].astype(str).str.contains(query)) |
                        (df_to_process['buyer_char'].astype(str).str.contains(query))
                    ]

            # session_state에서 필터 값 가져오기
            df_filtered = df_to_process[df_to_process['price'] >= st.session_state.min_price].copy()
            
            # session_state에서 기준 총 거래액 가져오기
            base_data, base_details = data_processing_by_price(
                df_filtered, 
                amount=st.session_state.amount_threshold
            )
            
            # Session State에 결과 저장
            st.session_state.base_edge_data = base_data
            st.session_state.df_filtered_original = df_filtered # 그래프용 원본
            st.session_state.base_detail_data = base_details # 상세 데이터 저장
            
            # 필터링용 노드 ID 리스트 생성
            if not base_data.empty:
                node_ids = pd.concat([
                    base_data['seller_account'], 
                    base_data['buyer_account']
                ]).unique()
                st.session_state.all_node_ids = sorted(list(node_ids))
            else:
                st.session_state.all_node_ids = []

    # --- UI 위젯 정의 ---
    st.sidebar.subheader("1. 그래프 구성")
    st.sidebar.number_input(
        "기준 총 거래액 (total_price >)", 
        min_value=0, value=st.session_state.amount_threshold, step=100000,
        help="이 금액을 초과하는 총 거래 관계를 대상으로 네트워크를 생성합니다.",
        key='amount_threshold'
    )
    st.sidebar.selectbox(
        "노드(원) 크기 기준", options=[NODE_SIZE_PRICE, NODE_SIZE_CONNECTION], index=[NODE_SIZE_PRICE, NODE_SIZE_CONNECTION].index(st.session_state.node_size_standard),
        help="노드 크기를 '총 거래액' 또는 '연결 수' 기준으로 결정합니다.",
        key='node_size_standard'
    )

    st.sidebar.button(
        "🚀 그래프 생성", 
        on_click=generate_graph_data,
        help="클릭 시 데이터를 처리하고 메인 화면에 그래프를 표시합니다."
    )

    st.sidebar.divider()

    st.sidebar.subheader("2. 데이터 필터링 (옵션)")
    st.sidebar.radio(
    "특정 계정/캐릭터 필터", 
    [FILTER_TYPE_ACCOUNT, FILTER_TYPE_CHAR],
    index=[FILTER_TYPE_ACCOUNT, FILTER_TYPE_CHAR].index(st.session_state.filter_type),
    help="특정 계정 또는 캐릭터와 관련된 거래만 필터링합니다.",
    key='filter_type'
    )
    st.sidebar.text_input(
        "Vopenid 또는 Vroleid 입력",
        placeholder="전체 또는 일부 입력",
        key='filter_value'
    )
    st.sidebar.number_input(
        "최소 개별 거래액", 
        min_value=0, value=st.session_state.min_price, step=1000,
        help="이 금액 미만인 개별 거래는 최초 데이터에서 제외합니다.",
        key='min_price'
    )

    def display_graph(node_count, selected_account):
        """네트워크 그래프를 조건에 따라 표시합니다."""
        st.subheader("📈 네트워크 그래프")
        
        if node_count > MAX_NODES_TO_RENDER and not st.session_state.get('force_render', False):
            st.error(f"⚠️ **성능 경고:** 시각화할 노드의 개수({node_count}개)가 너무 많습니다.")
            if st.button("그래도 그래프 생성하기 (앱이 멈출 수 있습니다)"):
                st.session_state.force_render = True
                st.rerun()
            st.warning(f"느린 속도를 원치 않으시면, 사이드바의 '기준 총 거래액'을 높여 노드 개수를 {MAX_NODES_TO_RENDER}개 이하로 줄여주세요.")
            return
    
        if selected_account == "-- 전체 보기 --":
            display_edge_data = st.session_state.base_edge_data
            title_text = f"전체 거래 네트워크 (기준금액: {st.session_state.amount_threshold:,.0f})"
        else:
            display_edge_data = search_df(st.session_state.base_edge_data, selected_account)
            title_text = f"'{selected_account}' 계정 거래 네트워크"
        
        if display_edge_data.empty:
            st.warning("선택한 조건에 맞는 그래프 데이터가 없습니다.")
        else:
            fig, adj_list_json = network_graph(
                display_edge_data, 
                st.session_state.df_filtered_original,
                title_text=title_text, 
                standard=st.session_state.node_size_standard
            )
            
            graph_json = fig.to_json()
            js_script = f'''
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <div id="plotly-graph-div"></div>
            <script>
                var spec = {graph_json};
                var adj = {adj_list_json};
                var graphDiv = document.getElementById('plotly-graph-div');
                Plotly.newPlot(graphDiv, spec.data, spec.layout);
    
                // --- Clipboard copy logic ---
                function fallbackCopyToClipboard(text) {{
                    var textArea = document.createElement("textarea");
                    textArea.value = text;
                    textArea.style.top = "0"; textArea.style.left = "0"; textArea.style.position = "fixed";
                    document.body.appendChild(textArea);
                    textArea.focus(); textArea.select();
                    try {{
                        var successful = document.execCommand('copy');
                        if (successful) alert('계정 ID가 클립보드에 복사되었습니다: ' + text);
                        else alert('클립보드 복사에 실패했습니다.');
                    }} catch (err) {{
                        console.error('Fallback clipboard copy failed:', err);
                        alert('클립보드 복사에 실패했습니다.');
                    }}
                    document.body.removeChild(textArea);
                }}
                function copyToClipboard(text) {{
                    if (navigator.clipboard && window.isSecureContext) {{
                        navigator.clipboard.writeText(text).then(function() {{
                            alert('계정 ID가 클립보드에 복사되었습니다: ' + text);
                        }}, function(err) {{
                            fallbackCopyToClipboard(text);
                        }});
                    }} else {{
                        fallbackCopyToClipboard(text);
                    }}
                }}
    
                graphDiv.on('plotly_click', function(data) {{
                    if (data.points.length > 0) {{
                        var point = data.points[0];
                        // curveNumber 1 is the node_trace
                        if (point.curveNumber === 1 && point.customdata) {{
                            copyToClipboard(point.customdata);
                        }}
                    }}
                }});
    
                // --- Highlighting logic ---
                graphDiv.on('plotly_hover', function(data) {{
                    if (data.points.length > 0) {{
                        var point = data.points[0];
                        // curveNumber 1 is the node_trace
                        if (point.curveNumber === 1) {{
                            var pointNumber = point.pointNumber;
                            var neighbors = adj[pointNumber];
                            
                            var numNodes = spec.data[1].x.length;
                            var opacities = Array(numNodes).fill(0.2);
                            
                            opacities[pointNumber] = 1.0;
                            neighbors.forEach(function(neighborIdx) {{
                                opacities[neighborIdx] = 1.0;
                            }});
                            
                            Plotly.restyle(graphDiv, {{'marker.opacity': [opacities]}}, [1]);
                        }}
                    }}
                }});
    
                graphDiv.on('plotly_unhover', function(data) {{
                    Plotly.restyle(graphDiv, {{'marker.opacity': 1}}, [1]);
                }});
            </script>
            '''
            components.html(js_script, height=800, scrolling=False)    
    def display_table(selected_account):
        """상세 거래 데이터 테이블을 표시합니다."""
        st.subheader("📊 상세 거래 데이터")
        
        if selected_account == "-- 전체 보기 --":
            display_detail_data = st.session_state.base_detail_data
        else:
            display_detail_data = st.session_state.base_detail_data[
                (st.session_state.base_detail_data['seller_account'] == selected_account) |
                (st.session_state.base_detail_data['buyer_account'] == selected_account)
            ]
    
        st.write("테이블에 표시할 컬럼을 선택하세요:")
        all_possible_cols = ['izoneareaid', 'sell_time', 'seller_account', 'seller_char', 'seller_lv', 'auction_no', 'price', 'item_index', 'item_no', 'seller 총과금액', 'buy_time', 'buyer_account', 'buyer_char', 'buyer_lv', 'tier', 'gear_score', 'buyer 총과금액', 'soul_index', 'item_extra_option', '가위횟수', '스타포스레벨', '장비레벨', '초월레벨', '문장인덱스', '아이템명', '소울']
        default_cols = ['sell_time', 'seller_account', 'buyer_account', 'price', 'gear_score', '아이템명', '가위횟수', '스타포스레벨', '장비레벨', '초월레벨', '문장인덱스', '소울']
    
        if not display_detail_data.empty:
            available_cols_in_order = [col for col in all_possible_cols if col in display_detail_data.columns]
            default_cols_in_order = [col for col in default_cols if col in available_cols_in_order]
            selected_cols = st.multiselect(
                label="표시할 컬럼 선택",
                options=available_cols_in_order,
                default=default_cols_in_order,
                label_visibility="collapsed"
            )
            
            if selected_cols:
                df_to_show = display_detail_data[selected_cols].copy()
                rename_dict = {'sell_time': '판매시간', 'buy_time': '구매시간', 'price': '거래가격'}
                df_to_show.rename(columns={k: v for k, v in rename_dict.items() if k in df_to_show.columns}, inplace=True)
                if '거래가격' in df_to_show.columns:
                    df_to_show.sort_values(by="거래가격", ascending=False, inplace=True)
                st.dataframe(df_to_show)

                csv_data = df_to_show.to_csv(index=False).encode('utf-8-sig')
                file_name = f'detail_{selected_account}.csv' if selected_account != "-- 전체 보기 --" else "detail_all.csv"
                st.download_button(
                    label="📥 CSV로 다운로드",
                    data=csv_data,
                    file_name=file_name,
                    mime='text/csv',
                )
            else:
                st.warning("표시할 컬럼을 하나 이상 선택해주세요.")
        else:
            st.info("표시할 상세 거래 데이터가 없습니다.")
    
    def display_main_content():
        """메인 콘텐츠(그래프, 테이블 등)를 표시합니다."""
        if st.session_state.base_edge_data is None:
            st.info("사이드바에서 옵션을 설정한 후 '그래프 생성' 버튼을 눌러주세요.")
            return
    
        # --- 성능 안전장치 & 계정 필터 ---
        node_count = pd.concat([st.session_state.base_edge_data['seller_account'], st.session_state.base_edge_data['buyer_account']]).nunique()
    
        st.subheader("🔍 계정 ID로 필터링")
        st.caption("그래프와 하단 테이블에 모두 적용됩니다.")
        filter_options = ["-- 전체 보기 --"] + st.session_state.all_node_ids
        selected_account = st.selectbox(
            "필터링할 계정 ID를 선택하세요:",
            options=filter_options,
            index=0,
            label_visibility="collapsed"
        )
    
        display_graph(node_count, selected_account)
        display_table(selected_account)
    
    # --- 메인 화면 ---
    display_main_content()
    
    


