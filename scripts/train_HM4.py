# Training script for HM4 model
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from expected_cost.ec import average_cost
from expected_cost.ec import CostMatrix
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader

import hm4.preprocessing as classifier_preproc
from hm4.expected_cost import COST_ARRAY
from hm4.VAE_classifier_248 import VAEClassifier

N_CLASSES = 38
PATIENCE = 50

parser = argparse.ArgumentParser(description="""Train a VAE on CRISM data.""")

parser.add_argument(
    "--train_data",
    type=str,
    required=True,
    help="Filepath to the ratioed training data.",
)

parser.add_argument(
    "--val_data",
    type=str,
    required=True,
    help="Filepath to the ratioed validation data."
)

parser.add_argument(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Name of the model to save.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Output directory to save the results to."
)

parser.add_argument(
    "--n_blocks",
    type=int,
    default=3,
    help="""Number of upsampling/downsampling blocks in the encoder.
    Default is 3.""",
)

parser.add_argument(
    "--n_conv_layers",
    type=int,
    default=1,
    help="""Number of convolutional layers in each block. Default is 1.""",
)

parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for training the VAE. Default is 64.",
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train the VAE. Default is 100.",
)

parser.add_argument(
    "-zx",
    "--zx_dim",
    type=int,
    default=16,
    help="Dimension of the entangled latent space. Default is 16.",
)

parser.add_argument(
    "-zy",
    "--zy_dim",
    type=int,
    default=16,
    help="Dimension of the disentangled latent space. Default is 16.",
)

parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate for the Adam optimizer. Default is 1e-2.",
)

parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=75000,
    help="Weight of the classification loss. Default is 75000.",
)

arguments = parser.parse_args()
args = vars(arguments)

# print all args and values
for arg, value in args.items():
    print(f"{arg} : {value} \t", flush=True)
print("\n")

output_dir = os.path.join(args["output_dir"], args["name"])
if os.path.exists(output_dir):
    raise ValueError(
        "Output directory already exists. Please choose a new name."
    )

os.makedirs(output_dir)

train_data = classifier_preproc.CRISMData(
    args["train_data"],
    transform=True,
    bands_to_use=(0, 248),
)
val_data = classifier_preproc.CRISMData(
    args["val_data"],
    transform=True,
    bands_to_use=(0, 248),
)

train_loader = DataLoader(
    train_data, batch_size=args["batch_size"], shuffle=True
)
val_loader = DataLoader(val_data, batch_size=args["batch_size"], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAEClassifier(
    n_blocks=args["n_blocks"],
    n_conv_layers=args["n_conv_layers"],
    zx_dim=args["zx_dim"],
    zy_dim=args["zy_dim"],
    n_classes=N_CLASSES,
).to(device)
optimizer = Adam(model.parameters(), lr=args["learning_rate"])

train_losses = []
train_re_x_losses = []
train_re_y_losses = []
train_kl_x_losses = []
train_kl_y_losses = []
train_acc_list = []
train_ec_list = []
val_losses = [1000.0]
val_re_x_losses = []
val_re_y_losses = []
val_kl_x_losses = []
val_kl_y_losses = []
val_acc_list = []
val_ec_list = []
best_model = None

beta_start = np.zeros(20)
beta_ramp = np.arange(0, 1, 1 / 50)
beta_end = np.ones(args["epochs"] - len(beta_start) - len(beta_ramp))
beta = np.concatenate((beta_start, beta_ramp, beta_end))

gamma = args["gamma"]  # Weight of the classification loss

cost_matrix = CostMatrix(COST_ARRAY)
priors = np.full(38, 1 / 38)  # Uniform prior

for epoch in range(args["epochs"]):
    model.train()
    train_loss = 0
    re_x_loss = 0
    re_y_loss = 0
    kl_x_loss = 0
    kl_y_loss = 0
    train_acc = []
    train_ec = 0
    for x, y in train_loader:
        x = x.unsqueeze(1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        x_re, y_re, x_kl, y_kl, y_recon = model.loss_function(x, y)
        # take mean of KL div across batch, to match the reconstruction losses
        x_kl = x_kl.mean()
        y_kl = y_kl.mean()
        # Multiply KL div by beta and invert as we want to minimise the KL div
        x_kl = -(x_kl * beta[epoch])
        y_kl = -(y_kl * beta[epoch])
        # Multiply the classification loss by gamma
        y_re = gamma * y_re
        x_re = (gamma / 3) * x_re
        # Total loss
        loss = x_re + y_re + x_kl + y_kl
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # Calculate classification metrics
        y_recon = y_recon.to("cpu").detach().numpy()
        y = y.to("cpu").detach().numpy()
        y_pred = np.argmax(y_recon, axis=1)
        train_acc.append(accuracy_score(y, y_pred))
        train_ec += average_cost(y, y_pred, cost_matrix, priors)
        train_loss += loss.item()
        re_x_loss += x_re.item()
        re_y_loss += y_re.item()
        kl_x_loss += x_kl.item()
        kl_y_loss += y_kl.item()
    train_losses.append(train_loss / len(train_loader))
    train_re_x_losses.append(re_x_loss / len(train_loader))
    train_re_y_losses.append(re_y_loss / len(train_loader))
    train_kl_x_losses.append(kl_x_loss / len(train_loader))
    train_kl_y_losses.append(kl_y_loss / len(train_loader))
    train_acc_list.append(np.array(train_acc).mean())
    train_ec_list.append(train_ec / len(train_loader))

    model.eval()
    val_loss = 0
    re_x_loss = 0
    re_y_loss = 0
    kl_x_loss = 0
    kl_y_loss = 0
    val_acc = []
    val_ec = 0
    for x, y in val_loader:
        x = x.unsqueeze(1).to(device)
        y = y.to(device)
        x_re, y_re, x_kl, y_kl, y_recon = model.loss_function(x, y)
        x_kl = x_kl.mean()
        y_kl = y_kl.mean()
        x_kl = -(x_kl * beta[epoch])
        y_kl = -(y_kl * beta[epoch])
        y_re = gamma * y_re
        loss = x_re + y_re + x_kl + y_kl
        y_recon = y_recon.to("cpu").detach().numpy()
        y = y.to("cpu").detach().numpy()
        y_pred = np.argmax(y_recon, axis=1)
        val_acc.append(accuracy_score(y, y_pred))
        val_ec += average_cost(y, y_pred, cost_matrix, priors)
        val_loss += loss.item()
        re_x_loss += x_re.item()
        re_y_loss += y_re.item()
        kl_x_loss += x_kl.item()
        kl_y_loss += y_kl.item()
    val_losses.append(val_loss / len(val_loader))
    val_re_x_losses.append(re_x_loss / len(val_loader))
    val_re_y_losses.append(re_y_loss / len(val_loader))
    val_kl_x_losses.append(kl_x_loss / len(val_loader))
    val_kl_y_losses.append(kl_y_loss / len(val_loader))
    val_acc_list.append(np.array(val_acc).mean())
    val_ec_list.append(val_ec / len(val_loader))

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]},"
        f" Val Loss: {val_losses[-1]}",
        flush=True,
    )

    if epoch % 10 == 0:
        fig, ax = plt.subplots(3, 3, figsize=(15, 10))
        ax[0, 0].plot(train_losses, label="Train")
        ax[0, 0].plot(val_losses[1:], label="Val")
        ax[0, 0].set_title("Total loss")
        ax[0, 0].legend()
        ax[0, 1].plot(train_re_x_losses, label="Train")
        ax[0, 1].plot(val_re_x_losses, label="Val")
        ax[0, 1].set_title("Reconstruction loss for X")
        ax[0, 1].legend()
        ax[0, 2].axis("off")
        ax[1, 0].plot(train_kl_x_losses, label="Train")
        ax[1, 0].plot(val_kl_x_losses, label="Val")
        ax[1, 0].set_title("KL divergence for X")
        ax[1, 0].legend()
        ax[1, 1].plot(train_kl_y_losses, label="Train")
        ax[1, 1].plot(val_kl_y_losses, label="Val")
        ax[1, 1].set_title("KL divergence for Y")
        ax[1, 1].legend()
        ax[1, 2].plot(beta[: epoch + 1])
        ax[1, 2].set_title("Beta")
        ax[2, 0].plot(train_re_y_losses, label="Train")
        ax[2, 0].plot(val_re_y_losses, label="Val")
        ax[2, 0].set_title("Reconstruction loss for Y (classification)")
        ax[2, 0].legend()
        ax[2, 1].plot(train_acc_list, label="Train")
        ax[2, 1].plot(val_acc_list, label="Val")
        ax[2, 1].set_title("Classification Accuracy")
        ax[2, 1].legend()
        ax[2, 2].plot(train_ec_list, label="Train")
        ax[2, 2].plot(val_ec_list, label="Val")
        ax[2, 2].set_title("Classification Expected Cost")
        ax[2, 2].legend()
        plt.savefig(os.path.join(output_dir, f"{args['name']}_loss.png"))
        plt.clf()

    if epoch % 50 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                output_dir, f"{args['name']}_epoch{epoch}_weights.pth"
            ),
        )

    # Only start looking at early stopping after all loss
    # annealing has been completed
    if epoch > 120:
        if val_losses[-1] < min(
            val_losses[(args["epochs"]) - len(beta_end): -1]
        ):
            PATIENCE = 50
            best_model = model
            best_epoch = epoch
        else:
            PATIENCE -= 1
        if PATIENCE == 0:
            print("Early stopping")
            torch.save(
                model.state_dict(),
                os.path.join(
                    output_dir, f"{args['name']}_epoch{epoch}_weights.pth"
                ),
            )
            break
    if epoch == 0:
        val_losses.pop(0)
    if np.isnan(train_losses[-1]):
        print("Nan loss")
        if best_model is not None:
            torch.save(
                best_model.state_dict(),
                os.path.join(
                    output_dir,
                    f"{args['name']}_epoch{best_epoch}_best_weights.pth"
                ),
            )
        break

if best_model is not None:
    torch.save(
        best_model.state_dict(),
        os.path.join(
            output_dir, f"{args['name']}_epoch{best_epoch}_best_weights.pth"
        ),
    )
