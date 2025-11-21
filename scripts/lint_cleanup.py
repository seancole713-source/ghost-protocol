#!/usr/bin/env python3
"""
Automated Python Lint Cleanup Script

Fixes common lint issues:
- Remove unused imports
- Fix asyncio.TimeoutError → TimeoutError
- Remove unnecessary f-strings without placeholders
- Remove trailing whitespace
- Remove unused variables (safe cases only)
"""

import re
import sys


def cleanup_wolf_app():
    """Clean up wolf_app.py lint issues."""
    with open('wolf_app.py', 'r') as f:
        content = f.read()

    original = content
    changes = []

    # Fix 1: Remove unused 'random' import
    if '\nimport random\n' in content:
        content = content.replace('\nimport random\n', '\n')
        changes.append("Removed unused 'random' import")

    # Fix 2: Replace asyncio.TimeoutError with TimeoutError
    count = content.count('asyncio.TimeoutError')
    content = re.sub(
        r'except \(asyncio\.TimeoutError, TimeoutError\)',
        'except TimeoutError',
        content
    )
    content = re.sub(
        r'except asyncio\.TimeoutError',
        'except TimeoutError',
        content
    )
    if count > 0:
        changes.append(f"Replaced {count} asyncio.TimeoutError with TimeoutError")

    # Fix 3: Remove unused fastapi.requests.Request import
    pattern = r'from fastapi\.requests import Request\n'
    if re.search(pattern, content):
        content = re.sub(pattern, '', content)
        changes.append("Removed unused fastapi.requests.Request import")

    # Fix 4: Remove unused price_reliability imports
    pattern = r'from core\.price_reliability import get_price_with_fallback, get_provider_stats\n'
    if re.search(pattern, content):
        content = re.sub(pattern, '', content)
        changes.append("Removed unused price_reliability imports")

    # Fix 5: Fix f-strings without placeholders (prediction check section)
    content = re.sub(
        r'msg = f"(⚠️ PREDICTION CHECK\\n\\n)"',
        r'msg = "\1"',
        content
    )
    content = re.sub(
        r'msg \+= f"(PREDICTED:|ACTUAL:)\\n"',
        r'msg += "\1\\n"',
        content
    )
    if 'msg = "⚠️ PREDICTION CHECK' in content:
        changes.append("Fixed f-strings without placeholders")

    # Fix 6: Remove unused Exception variable 'e'
    content = re.sub(
        r'except Exception as e:\n(\s+)pass  # Continue',
        r'except Exception:\n\1pass  # Continue',
        content
    )
    changes.append("Removed unused Exception variables")

    # Fix 7: Remove unused 'prev' variable
    content = re.sub(
        r'(\n\s+)prev = None\n',
        r'\n',
        content
    )

    # Fix 8: Remove unused crypto_provider_health
    content = re.sub(
        r'(\n\s+)crypto_provider_health = \{\}\n',
        r'\n',
        content
    )

    # Fix 9: Remove trailing whitespace
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    changes.append("Removed trailing whitespace")

    if content != original:
        with open('wolf_app.py', 'w') as f:
            f.write(content)
        
        print("✅ LINT CLEANUP COMPLETE\n")
        print("Changes made:")
        for i, change in enumerate(changes, 1):
            print(f"  {i}. {change}")
        print(f"\nTotal fixes applied: {len(changes)}")
        return True
    else:
        print("✅ No changes needed")
        return False


if __name__ == "__main__":
    try:
        success = cleanup_wolf_app()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)
