import re
import glob
import numpy as np

def summarize(files):
    if len(files) < 5:
        print("There is no results for this set.")
        return 0,0

    dices = []
    for file in files[:5]:
        fd = open(file)
        lines = fd.readlines()
        # print(len(lines),file)
        result_line = lines[-2]
        find = re.search("0.*",result_line)
        line_parts = re.split(" ", result_line)
        dice = line_parts[-2]
        dices.append(float(dice))

    return dices

def main():
    bcv_dir = "runs/log/bcv"
    decathlon_dir = "runs/log/decathlon"
    ctorg_dir = "runs/log/ctorg"
    dir = "runs/log/ctorg_5shot"
    dirs = ["runs/log/ctorg_5shot", "runs/log/decathlon_5shot"]

    for dir in dirs:
        print()
        print(dir)
        for shot in [1,3,5]:
            for organ in [1,3,6,14]:
                files = glob.glob(f"{dir}/*{shot}shot*_{organ}_*")
                files.sort()
                dices = summarize(files)
                avg, std = np.mean(dices), np.std(dices)
                avg = float("{:.3f}".format(avg))
                std = float("{:.4f}".format(std))
                dices = [str(dice) for dice in dices]
                dice_str = ",".join(dices)
                # print(dir, organ, shot)
                if avg*std!=0:
                    print(f"organ:{organ},{shot}shot,{dice_str},{avg},{std}")

        # assert False


if __name__=="__main__":
    main()
