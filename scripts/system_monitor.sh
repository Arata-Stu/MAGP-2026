#!/bin/bash
# ============================================
# CUI Dashboard for RC Car (Full tmux Support)
# ============================================

show_menu() {
    cmd=(dialog --checklist "起動する項目を選択 (スペースで複数選択):" 22 70 15)
    options=(
        1 "LaserScan (BEV)" off
        2 "System Monitor (CPU/Mem)" off
        3 "Topic HZ Monitor" off
        4 "Param Loader (tmux内)" off
        5 "Jetson jtop Monitor" off
        0 "終了 (Exit)" off
    )
    choices=$("${cmd[@]}" "${options[@]}" 2>&1 >/dev/tty)
    echo $choices
}

launch_topic_hz_monitor_wrapper() {
    local SCRIPT_DIR=$(dirname "$0")
    local PY_SCRIPT_PATH="$SCRIPT_DIR/py/topic_hz_monitor.py"
    dialog --infobox "ROS2トピックを検索中..." 4 30
    topic_list=$(ros2 topic list -t | grep -v "/parameter_events" | grep -v "/rosout" | awk '{gsub(/ \[|\]/, ""); print $1, $2, "off"}')
    if [ -z "$topic_list" ]; then
        dialog --msgbox "ROS2トピックが見つかりません。\nros2 daemonが起動しているか確認してください。" 10 50
        return
    fi
    selected_topics=$(dialog --checklist "監視するトピックを選択 (スペースキー)" 25 70 18 $topic_list 2>&1 >/dev/tty)
    if [ $? -ne 0 ] || [ -z "$selected_topics" ]; then
        dialog --msgbox "トピックが選択されませんでした。" 5 40
        return
    fi
    echo "$topic_list" > /tmp/dashboard_topics.txt
    py_args=()
    for topic_name in $selected_topics; do
        topic_name_clean=$(echo $topic_name | tr -d '"')
        msg_type=$(grep "^$topic_name_clean " /tmp/dashboard_topics.txt | awk '{print $2}')
        if [ ! -z "$msg_type" ]; then
            py_args+=("${topic_name_clean},${msg_type}")
        fi
    done
    rm /tmp/dashboard_topics.txt
    if [ ${#py_args[@]} -eq 0 ]; then
         dialog --msgbox "トピックの解析に失敗しました。" 5 40
         return
    fi
    python3 "$PY_SCRIPT_PATH" "${py_args[@]}"
}

launch_param_loader_dialog() {
    local current_dir="." 
    while true; do
        dialog --infobox "ROS2ノードを検索中..." 4 30
        node_list=$(ros2 node list | grep -v "/_ros2cli_" | grep -v "^/$" | awk '{print NR, $1}' | tr '\n' ' ')
        if [ -z "$node_list" ]; then
            dialog --msgbox "ROS2ノードが見つかりません。\n(Ctrl+Cでこのペインを終了)" 10 50
            sleep 5
            continue
        fi
        node_list="$node_list 0 終了(ペインを閉じる)"
        local node_tag=$(dialog --title "Param Loader" --menu "1. パラメータをロードするノードを選択" 20 60 15 $node_list 2>&1 >/dev/tty)
        if [ $? -ne 0 ] || [ "$node_tag" == "0" ]; then
            break
        fi
        local node_name=$(echo $node_list | tr ' ' '\n' | grep "^$node_tag$" -A 1 | tail -n 1)
        local yaml_file=$(dialog --title "Param Loader: $node_name" --fselect "$current_dir" 20 60 2>&1 >/dev/tty)
        if [ $? -ne 0 ]; then
            continue
        fi
        current_dir=$(dirname "$yaml_file")
        dialog --infobox "実行中: \nros2 param load $node_name $yaml_file" 6 70
        local result=$(ros2 param load "$node_name" "$yaml_file" 2>&1)
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            dialog --title "成功" --msgbox "✅ パラメータのロードに成功しました。\n\n$result" 15 70
        else
            dialog --title "エラー" --msgbox "❌ パラメータのロードに失敗しました。\n\n$result" 15 70
        fi
    done
    clear
    echo "Param Loader (dialog) を終了しました。"
}

launch_jtop_monitor() {
    if ! command -v jtop &> /dev/null; then
        dialog --msgbox "エラー: jtop がインストールされていません。\n\nsudo apt install python3-jtop\nでインストールしてください。" 10 60
        return
    fi
    clear
    echo "Jetson jtop モニタを起動します。終了は 'q' キーです。"
    sleep 1
    sudo jtop
}

SCRIPT_DIR=$(dirname "$0")
PY_DIR="$SCRIPT_DIR/py"

if [ "$1" == "--run-scan" ]; then
    python3 "$PY_DIR/scan_viewer.py"
    exit 0
fi
if [ "$1" == "--run-sys" ]; then
    python3 "$PY_DIR/sys_monitor.py"
    exit 0
fi
if [ "$1" == "--run-topic-hz" ]; then
    launch_topic_hz_monitor_wrapper
    exit 0
fi
if [ "$1" == "--run-param-loader" ]; then
    launch_param_loader_dialog
    exit 0
fi
if [ "$1" == "--run-jtop" ]; then
    launch_jtop_monitor
    exit 0
fi

for cmd in dialog tmux ros2 python3; do
    if ! command -v $cmd &> /dev/null; then
        echo "エラー: '$cmd' コマンドが見つかりません。" >&2
        exit 1
    fi
done
for lib in psutil; do
    if ! python3 -c "import $lib" &> /dev/null; then
        echo "エラー: Pythonライブラリ '$lib' が見つかりません。" >&2
        echo "'pip3 install $lib' を実行してください。" >&2
        exit 1
    fi
done
for py_file in "scan_viewer.py" "sys_monitor.py" "topic_hz_monitor.py"; do
    if [ ! -f "$PY_DIR/$py_file" ]; then
        echo "エラー: 必要なファイル '$py_file' が見つかりません。" >&2
        echo "'$PY_DIR' ディレクトリに配置してください。" >&2
        exit 1
    fi
done

choices=$(show_menu)
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "メニューがキャンセルされました。"
    exit 0
fi

clear
choices_clean=$(echo $choices | tr -d '"')
has_0=0; has_1=0; has_2=0; has_3=0; has_4=0; has_5=0
for c in $choices_clean; do
    if [ "$c" == "0" ]; then has_0=1; fi
    if [ "$c" == "1" ]; then has_1=1; fi
    if [ "$c" == "2" ]; then has_2=1; fi
    if [ "$c" == "3" ]; then has_3=1; fi
    if [ "$c" == "4" ]; then has_4=1; fi
    if [ "$c" == "5" ]; then has_5=1; fi
done

if [ $has_0 -eq 1 ]; then
    clear
    echo "ダッシュボードを終了します。"
    exit 0
fi

SCRIPT_PATH="$0"
SESSION_NAME="cui_dashboard_$$"

commands_to_run=()
if [ $has_1 -eq 1 ]; then commands_to_run+=("bash $SCRIPT_PATH --run-scan"); fi
if [ $has_2 -eq 1 ]; then commands_to_run+=("bash $SCRIPT_PATH --run-sys"); fi
if [ $has_3 -eq 1 ]; then commands_to_run+=("bash $SCRIPT_PATH --run-topic-hz"); fi
if [ $has_4 -eq 1 ]; then commands_to_run+=("bash $SCRIPT_PATH --run-param-loader"); fi
if [ $has_5 -eq 1 ]; then commands_to_run+=("bash $SCRIPT_PATH --run-jtop"); fi

num_commands=${#commands_to_run[@]}
if [ $num_commands -eq 0 ]; then
    echo "モニタリング項目が選択されませんでした。"
else
    echo "tmuxセッション ($SESSION_NAME) を起動します..."
    echo "(セッションから抜ける(デタッチ)には Ctrl+B -> D)"
    sleep 1
    tmux new-session -d -s "$SESSION_NAME" "${commands_to_run[0]}"
    for (( i=1; i<num_commands; i++ )); do
        tmux split-window -h -t "$SESSION_NAME:0" "${commands_to_run[$i]}"
        tmux select-layout -t "$SESSION_NAME:0" tiled 
    done
    tmux attach-session -t "$SESSION_NAME"
    echo "セッション $SESSION_NAME をクリーンアップします。"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
fi

clear
echo "ダッシュボードを終了します。"
