import argparse, time
from run_batch import run_full_ocr_pipeline_from_cache
from pipeline import _PaddleEngine  # dùng để tạo 1 engine dùng chung (tăng tốc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pdf", action="store_true", help="Bù text từ PDF vector (ăn xổi).")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--rec_batch", type=int, default=32, help="Batch size recognizer (tăng tốc, không đổi kết quả).")
    args = ap.parse_args()

    # 1 engine cho cả batch -> tránh init lại, nhanh hơn
    eng = _PaddleEngine(use_gpu=args.gpu, rec_batch_num=args.rec_batch)

    t0 = time.perf_counter()
    res = run_full_ocr_pipeline_from_cache(
        args.cache,
        use_pdf_validate=args.pdf,
        paddle_kwargs={"use_gpu": args.gpu, "rec_batch_num": args.rec_batch},
        shared_engine=eng,
    )
    print(f"[DONE] images={len(res)} | total={time.perf_counter()-t0:.2f}s")
