import os
import sys
import shutil
from tensorboardX import SummaryWriter


def load_lr_loss(log_dir):
    with open(os.path.join(log_dir, "log.txt")) as f:
        lines = f.readlines()
    epoch = -1
    lr = -0.0001
    train_loss = -0.0001
    val_loss = -0.0001
    for idx, line in reversed(list(enumerate(lines))):
        if line.find("Train. Epoch: 0") != -1:
            lines = lines[idx-1:]
            break
    results = []
    for line in lines:
        line = line.strip()
        if line.find("LR:") == 0:
            lr = float(line.split(":")[1])
        elif line.find("[RESULT]: Train. Epoch:") != -1:
            tokens = line.split(",")
            epoch = int(tokens[0].replace("[RESULT]: Train. Epoch: ", ""))
            train_loss = float(tokens[1].replace("summary_loss: ", ""))
        elif line.find("[RESULT]: Val. Epoch: ") != -1:
            tokens = line.split(",")
            val_loss = float(tokens[1].replace("summary_loss: ", ""))
            results.append((epoch+1, lr, train_loss, val_loss))
            print(results[-1])
            epoch = -1
            lr = -0.0001
            train_loss = -0.0001
            val_loss = -0.0001
    return results

def load_mAP(log_dir):
    with open(os.path.join(log_dir, "mAP.txt")) as f:
        lines = f.readlines()
    for idx, line in reversed(list(enumerate(lines))):
        if line.find("000epoch.bin") != -1:
            lines = lines[idx:]
            break

    all_APs = []
    APs = [-0.0001] * 11
    c = 0
    epoch = -1
    for line in lines:
        line = line.strip()
        if line.find("SUMMARY:") == 0:
            c = 0
        elif line.find("DETAIL:") == 0:
            c += 1
        elif line.find("epoch.bin") != -1:
            epoch = int(line.split("best-checkpoint-")[-1].replace("epoch.bin", ""))
        if c == 4:
            APs = [float(s) for s in line.replace("DETAIL:| ", "").replace("% |", "").replace("  ", " ").replace("  ", " ").strip().split(" ")]
            all_APs.append((epoch+1, APs[-1], APs[:-1]))
            print(all_APs[-1])
    return all_APs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python log2tensorboard.py _models/z0012_tensorboard_test")
        log_path = "_models/z0008_only_5_new_dataset_including_VAID_aabb_patient2_80epochs_train0.95_seed167"
    else:
        log_path = sys.argv[1]

    write_folders = ["train", "val", "AP75", "AP80", "AP85", "AP90", "AP95", "AP95_50"]
    write_folders = [os.path.join(log_path, folder) for folder in write_folders]
    for folder in write_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    writers = [SummaryWriter(logdir=folder) for folder in write_folders]
    writer_train, writer_val, writer_mAP75, writer_mAP80, writer_mAP85, writer_mAP90, writer_mAP95, writer_mAP95_50 = writers


    writer_mAP95.add_scalar('AP', 0.0, 0)
    writer_mAP75.add_scalar('AP', 100.0, 0)
    results = load_mAP(log_path)
    for epoch, mAP, APs in results:
        writer_mAP95_50.add_scalar('AP', mAP, epoch)
        writer_mAP95.add_scalar('AP', APs[-1], epoch)
        writer_mAP90.add_scalar('AP', APs[-2], epoch)
        writer_mAP85.add_scalar('AP', APs[-3], epoch)
        writer_mAP80.add_scalar('AP', APs[-4], epoch)
        writer_mAP75.add_scalar('AP', APs[-5], epoch)

    results = load_lr_loss(log_path)
    for epoch, lr, train_loss, val_loss in results:
        writer_train.add_scalar('loss', train_loss, epoch)
        writer_train.add_scalar('lr', lr, epoch)
        writer_val.add_scalar('loss', val_loss, epoch)
    writer_train.add_scalar('loss', 0.0, epoch+3)
    writer_train.add_scalar('lr', 0.0, epoch+3)

    for writer in writers:
        writer.close()

    if (len(sys.argv) <= 2 or sys.argv[2] != "--skip_board"):
        os.system(f"/home/me/anaconda3/bin/tensorboard --logdir {log_path}")

