from rgbt import LasHeR

lasher = LasHeR()
lasher(
    tracker_name="HAFTrack",
    result_path="",
    bbox_type="ltwh"
)

for att in lasher.get_attr_list():
    seqs = getattr(lasher, att)
    pr, _ = lasher.PR("HAFTrack", seqs=seqs)
    npr, _ = lasher.NPR("HAFTrack", seqs=seqs)
    sr, _ = lasher.SR("HAFTrack", seqs=seqs)
    print(f"{att}: PR={pr:.4f}, NPR={npr:.4f}, SR={sr:.4f}")