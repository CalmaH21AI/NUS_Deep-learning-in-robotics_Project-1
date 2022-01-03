import frozenlake as fl

# set parameters
grid = (10,10)
start = (0,0)
wins = [(9,9)]
fails = None
# fails = [(9,5),(9,2),(9,0),(8,3),(8,1),(7,1),(6,9),(6,1),(5,8),(5,6),(5,0),(4,9),(4,8),(4,7),(4,0),(3,1),(3,0),(2,8),(2,5),(2,2),(1,8),(1,1),(0,8),(0,7),(0,5)]
lr = 0.1 # learning rate
dr = 0.99 # discount rate
eps = 0.4 # epsilon
hole_rate = 0.25
iteration = 100


# create "fl" class
# MC
mc2=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps, hole_rate=hole_rate)
if fails == None:
    mc2.get_grid() # create a random map if 'fails' is None
    fails = mc2.fails # save the map for comparison
# SS
ss2=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps, hole_rate=hole_rate)
# QL
ql2=fl.fl(grid=grid, start=start, wins=wins, fails=fails, lr=lr, dr=dr, eps=eps, hole_rate=hole_rate)


# learning
mc2.learning("MC", iteration)
ss2.learning("SS", iteration)
ql2.learning("QL", iteration)


# plotting
# line plot
fl.lplot(mc=mc2.epi_values, ss=ss2.epi_values, ql=ql2.epi_values)
# point plot
fl.pplot(mc2.epi_infos, mc2.algo)
fl.pplot(ss2.epi_infos, ss2.algo)
fl.pplot(ql2.epi_infos, ql2.algo)
# heatmap
fl.show_heatmap(mc2.Q, mc2.algo)
fl.show_heatmap(ss2.Q, ss2.algo)
fl.show_heatmap(ql2.Q, ql2.algo)