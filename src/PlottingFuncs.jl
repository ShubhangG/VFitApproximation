"""
This function is used to plot the training and testing errors accumulated during multiple iterations of training
"""
function plot_iters(train_err, test_err, supp_pts,num,cnt)
    pts = collect(range(1,length=length(train_err)))
    PyPlot.plot(pts,log10.(train_err),".-",label="training error")
    PyPlot.plot(pts,log10.(test_err),"-", label="testing error")
    xlabel("iterations")
    ylabel("Log of least square error")
    legend(bbox_to_anchor=(0.7,0.5))
    title("Training and testing error in approximating function (support points=$(supp_pts)) in Vfit")
    savefig("./plots/train_test_err_supp_$(supp_pts)_num_$(num)_in_$(cnt)counts.png")
    close("all")

    return (train_err, test_err)
end

function gd_error_plot(train_err, supp_pts,num,cnt)
    pts = collect(range(1,length=length(train_err)))
    PyPlot.plot(pts,log10.(train_err),".-",label="training error")
    #PyPlot.plot(pts,log10.(test_err),"-", label="testing error")
    xlabel("Gradient Descent iterations")
    ylabel("Log of least square error")
    legend(bbox_to_anchor=(0.7,0.5))
    title("Training error in approximating function (support points=$(supp_pts)) in Vfit")
    savefig("./plots/GD_train_test_err_supp_$(supp_pts)_num_$(num)_in_$(cnt)counts.png")
    close("all")

    #return (train_err, test_err)

end