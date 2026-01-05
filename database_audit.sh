#!/bin/bash
# Ghost Database Audit Script
# Outputs complete inventory of all databases and their contents

echo "=============================================="
echo "GHOST DATABASE AUDIT REPORT"
echo "Generated: $(date)"
echo "=============================================="
echo ""

echo "## SQLITE DATABASE FILES (27 found)"
echo ""

for db in $(find . -name "*.db" -type f 2>/dev/null | sort); do
    size=$(ls -lh "$db" 2>/dev/null | awk '{print $5}')
    tables=$(sqlite3 "$db" "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%';" 2>/dev/null | wc -l)
    
    if [ "$tables" -gt 0 ]; then
        echo "### $db"
        echo "Size: $size | Tables: $tables"
        echo ""
        echo "| Table | Row Count |"
        echo "|-------|-----------|"
        for table in $(sqlite3 "$db" "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%';" 2>/dev/null); do
            count=$(sqlite3 "$db" "SELECT COUNT(*) FROM $table;" 2>/dev/null)
            echo "| $table | $count |"
        done
        echo ""
    fi
done

echo ""
echo "## CODE ANALYSIS"
echo ""
echo "Total sqlite3.connect calls in Python files:"
grep -r "sqlite3.connect" --include="*.py" . 2>/dev/null | grep -v ".venv" | wc -l

echo ""
echo "Files with most SQLite connections:"
grep -r "sqlite3.connect" --include="*.py" . 2>/dev/null | grep -v ".venv" | sed 's/:.*//g' | sort | uniq -c | sort -rn | head -10
