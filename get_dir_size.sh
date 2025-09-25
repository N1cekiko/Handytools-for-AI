du -sh */ ./* 2>/dev/null | sort -rh | while read -r line; do
    size=$(echo $line | awk '{print $1}')
    name=$(echo $line | awk '{$1=""; print substr($0, 2)}')
    time=$(stat -c %y "$name" | cut -d ' ' -f1)
    echo -e "$size\t$time\t$name"
done
