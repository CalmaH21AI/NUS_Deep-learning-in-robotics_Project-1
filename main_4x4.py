import frozenlake as fl

# set parameters
grid = (4,4)
start = (0,0)
wins = [(3,3)]
fails = [(1,1), (1,3), (2,3), (3,0)]
lr = 0.1 # learning rate
dr = 0.9 # discount rate
eps = 0.1 # epsilon
iteration = 100


# create "fl" class
# MC
mc1=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps)
# SS
ss1=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps)
# QL
ql1=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps)


# learning
mc1.learning("MC", iteration)
ss1.learning("SS", iteration)
ql1.learning("QL", iteration)


# plotting
# line plot
fl.lplot(mc=mc1.epi_values, ss=ss1.epi_values, ql=ql1.epi_values)
# point plot
fl.pplot(mc1.epi_infos, mc1.algo)
fl.pplot(ss1.epi_infos, ss1.algo)
fl.pplot(ql1.epi_infos, ql1.algo)
# heatmap
fl.show_heatmap(mc1.Q, mc1.algo)
fl.show_heatmap(ss1.Q, ss1.algo)
fl.show_heatmap(ql1.Q, ql1.algo)