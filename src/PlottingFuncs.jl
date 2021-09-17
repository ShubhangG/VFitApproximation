"""
This function is used to plot the training and testing errors accumulated during multiple iterations of training
"""
function plot_iters(train_err, test_err, supp_pts,num)
    pts = collect(range(1,length=length(train_err)))
    PyPlot.plot(pts,log10.(train_err),".-",label="training error")
    PyPlot.plot(pts,log10.(test_err),"-", label="testing error")
    xlabel("iterations")
    ylabel("Log of least square error")
    legend(bbox_to_anchor=(0.7,0.5))
    title("Training and testing error in approximating Square Root (support points=5) in Vfit")
    savefig("./plots/train_test_err_supp_$(supp_pts)_num_$(num).png")
    close("all")

    return (train_err, test_err)
end
