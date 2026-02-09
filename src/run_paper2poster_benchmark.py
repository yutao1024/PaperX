# 该脚本用来执行 Paper2Poster-eval 的 benchmark 测试（包括三个评测）需要在paper2poster目录下运行

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    """Run a command (list form) with error checking."""
    print(">>>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Traverse subfolders in A; if B has same subfolder name then skip; "
                    "otherwise record name and run Paper2Poster-eval benchmarks."
    )
    parser.add_argument("--dir_a", required=True, help="Folder A path (contains many paper subfolders).")
    parser.add_argument("--dir_b", required=True, help="Folder B path (contains many subfolders to check).")
    parser.add_argument("--poster_method", default="P2S_generated_posters", help="Value for --poster_method.")
    parser.add_argument("--log_path", default="", help="Optional log file path. Default: <dir_a>/missing_in_B.txt")
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    args = parser.parse_args()

    dir_a = Path(args.dir_a).expanduser().resolve()
    dir_b = Path(args.dir_b).expanduser().resolve()

    if not dir_a.exists() or not dir_a.is_dir():
        raise FileNotFoundError(f"dir_a not found or not a directory: {dir_a}")
    if not dir_b.exists() or not dir_b.is_dir():
        raise FileNotFoundError(f"dir_b not found or not a directory: {dir_b}")

    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else (dir_a / "missing_in_B.txt")

    missing_names: list[str] = []
    skipped_names: list[str] = []

    # 遍历 A 下所有直接子文件夹（只取一级）
    subdirs_a = sorted([p for p in dir_a.iterdir() if p.is_dir()], key=lambda x: x.name)

    for paper_dir in subdirs_a:
        paper_name = paper_dir.name
        b_same = dir_b / paper_name

        if b_same.exists() and b_same.is_dir():
            skipped_names.append(paper_name)
            print(f"[SKIP] B has same subdir: {paper_name}")
            continue

        # B 不存在同名子文件夹：记录并执行三条命令
        missing_names.append(paper_name)
        print(f"[RUN ] B missing subdir: {paper_name}")

        base = [
            "python", "-m", "Paper2Poster-eval.eval_poster_pipeline",
            "--paper_name", paper_name,
            "--poster_method", args.poster_method,
        ]

        try:
            run_cmd(base + ["--metric=qa"], dry_run=args.dry_run)
            run_cmd(base + ["--metric=judge"], dry_run=args.dry_run)
            run_cmd(base + ["--metric=stats"], dry_run=args.dry_run)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed for paper_name='{paper_name}'. Return code: {e.returncode}")
            # 不中断整体流程：继续跑下一个
            continue

    # 写日志：记录哪些 paper_name 在 B 中缺失（因此已运行）
    header = f"# Generated at {datetime.now().isoformat(timespec='seconds')}\n"
    header += f"# dir_a: {dir_a}\n# dir_b: {dir_b}\n"
    header += f"# poster_method: {args.poster_method}\n"
    header += f"# total_A_subdirs: {len(subdirs_a)}\n"
    header += f"# missing_in_B_and_ran: {len(missing_names)}\n"
    header += f"# skipped_existing_in_B: {len(skipped_names)}\n\n"

    content = header + "\n".join(missing_names) + ("\n" if missing_names else "")
    log_path.write_text(content, encoding="utf-8")

    print("\n==== Summary ====")
    print(f"Total A subdirs: {len(subdirs_a)}")
    print(f"Skipped (exists in B): {len(skipped_names)}")
    print(f"Ran (missing in B): {len(missing_names)}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
