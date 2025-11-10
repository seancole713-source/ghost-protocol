#!/usr/bin/env python3
import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("wolf_app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
