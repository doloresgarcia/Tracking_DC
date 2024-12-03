import numpy as np 
import pandas as pd

def make_cuts_ct(sd_hgb_ct):
    mask_R = sd_hgb_ct["R"].values<0.05
    mask_number_unique_hits =  sd_hgb_ct["number_unique_hits"].values>3
    mask_theta_lower = sd_hgb_ct["theta"].values>(10/180*np.pi)
    mask_theta_higher = sd_hgb_ct["theta"].values<(170/180*np.pi)
    mask_delta_MC = sd_hgb_ct["delta_MC"].values>0.02
    mask_pt = sd_hgb_ct["true_showers_pt"].values>1
    mask_gen_status = sd_hgb_ct["gen_status"]==1
    total_mask_ct = mask_R*mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_gen_status*mask_number_unique_hits
    mask_delta_plot = mask_R*mask_theta_lower*mask_theta_higher*mask_pt*mask_gen_status*mask_number_unique_hits
    mask_no_R = mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_pt*mask_gen_status*mask_number_unique_hits
    total_mask_ct_all_partices = mask_R*mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_number_unique_hits
    reconstructable_ct = sd_hgb_ct[total_mask_ct]
    reco_R_ct  = sd_hgb_ct[mask_no_R]
    reco_delta_plot = sd_hgb_ct[mask_delta_plot]
    reconstructable_ct_all_p = sd_hgb_ct[total_mask_ct_all_partices]
    return reconstructable_ct,reco_R_ct, reco_delta_plot, reconstructable_ct_all_p


def make_cuts_sd_hgb(sd_hgb):
    mask_R = sd_hgb["R"].values<0.05
    mask_number_unique_hits =  sd_hgb["number_unique_hits"].values>3
    mask_theta_lower = sd_hgb["theta"].values>(10/180*np.pi)
    mask_theta_higher = sd_hgb["theta"].values<(170/180*np.pi)
    mask_delta_MC = sd_hgb["delta_MC"].values>0.02
    mask_pt = sd_hgb["true_showers_pt"].values>1
    mask_gen_status = sd_hgb["gen_status"]==1
    total_mask = mask_R*mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_gen_status*mask_number_unique_hits
    mask_delta_plot = mask_R*mask_theta_lower*mask_theta_higher*mask_pt*mask_gen_status*mask_number_unique_hits
    mask_no_R = mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_pt*mask_gen_status*mask_number_unique_hits
    total_mask_all_particles = mask_R*mask_theta_lower*mask_theta_higher*mask_delta_MC*mask_number_unique_hits
    reconstructable_ml = sd_hgb[total_mask]
    reconstructable_ml_all_p = sd_hgb[total_mask_all_particles]
    reco_R_ml   = sd_hgb[mask_no_R]
    reco_delta_plot_ml = sd_hgb[mask_delta_plot]
    return reconstructable_ml, reconstructable_ml_all_p, reco_R_ml, reco_delta_plot_ml


def create_eff_main(sd_hgb_ct):
    reconstructable_ct,reco_R_ct, reco_delta_plot, reconstructable_ct_all_p = make_cuts_ct(sd_hgb_ct)
    eff_dict_ct = create_eff_dic(reconstructable_ct,reco_R_ct,reco_delta_plot)
    return eff_dict_ct


def create_eff_dic(matched_, reco_r,reco_delta):
    df_id = matched_
    photons_dic = calculate_eff(df_id)
    photons_dic = calculate_eff_vertex(reco_r,photons_dic)
    photons_dic = calculate_eff_DeltaMC(reco_delta,photons_dic)
    return photons_dic


def calculate_eff(sd, log_scale=False):
    # if log_scale:
    bins = np.exp(np.arange(np.log(1e-1), np.log(50), 0.1))
    # else:
    #     bins = np.arange(0, 51, 2)
    #bins = [5e-3,0.1,1.80804241e-01,2.98095799e-01, 4.91476884e-01, 8.10308393e-01, 1.33597268e+00,2.20264658e+00, 3.63155027e+00, 5.98741417e+00, 9.87157710e+00,1.62754791e+01, 2.68337287e+01, 4.42413392e+01]
    bins = np.array(bins)
    eff = []
    eff_def1 = []
    eff_50 = []
    eff_75 = []
    energy_eff = []
    size_energy_bin = []
    errors = []
    errors_def1 = []
    number_of_hits = []
    number_of_hits_var = []
    total_showers_ = []
    purity = []
    purity_var = []
    number_of_hits_unique = []
    number_of_hits_var_unique = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.true_showers_pt.values <= bin_i1
        mask_below = sd.true_showers_pt.values > bin_i
        # this mask takes all E that have values (does not include fakes then)
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_showers_E.values)[mask]
        )
        mask_non_nan = ~np.isnan(sd.e_pred_and_truth.values)
        total_showers = len(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            #total_reconstructed_cld_crit = np.sum(sd["e_pred_and_truth"][mask].values>3)
            particle_purity = sd["e_pred_and_truth"][mask].values/sd["reco_showers_E"][mask].values
            track_purity = sd["e_pred_and_truth"][mask].values/sd["pred_showers_E"][mask].values
            more_than_4_hits = sd["pred_showers_E"][mask].values>3
            total_reconstructed_cld_crit_purity =  np.sum((track_purity>=0.75)*more_than_4_hits)
            total_reconstructed_cld_crit =  np.sum((particle_purity>=0.5)*(track_purity>=0.5)*more_than_4_hits)
      

            # print(total_reconstructed_cld_crit,total_showers )
            percentage_of_hits_MC = sd["e_pred_and_truth"][mask*mask_non_nan].values/sd["reco_showers_E"][mask*mask_non_nan].values
            percentage_of_hits_MC_unique = sd["number_unique_hits_reconstructed"][mask*mask_non_nan].values/sd["number_unique_hits"][mask*mask_non_nan].values
            purity_list =  sd["e_pred_and_truth"][mask*mask_non_nan].values/sd["pred_showers_E"][mask*mask_non_nan].values
            n_t_purity =  sd["e_pred_and_truth"][mask*mask_non_nan].values
            n_f = sd["pred_showers_E"][mask*mask_non_nan].values-n_t_purity
            sigma_t_purity = np.var(n_t_purity)
            sigma_f_purity = np.var(n_f)
            error_purity = (n_f/(n_t_purity+n_f)**2*sigma_t_purity)**2+(n_t_purity*sigma_f_purity/(n_t_purity+n_f)**2)**2
            N = len(error_purity)
            purity.append(np.mean(purity_list))
            purity_var.append(1/N*(np.sqrt(np.sum(error_purity))))
            number_of_hits.append(np.mean(percentage_of_hits_MC))
            number_of_hits_unique.append(np.mean(percentage_of_hits_MC_unique))

            n_r = sd["e_pred_and_truth"][mask*mask_non_nan].values
            n_t = sd["reco_showers_E"][mask*mask_non_nan].values
            error_percentage_of_hits = (n_r/(n_t**2)*np.sqrt(n_t))**2+(1/n_t*np.sqrt(n_r))**2
            N = len(error_percentage_of_hits)
            number_of_hits_var.append(1/N*np.sqrt(np.sum(error_percentage_of_hits)))

            n_r = sd["number_unique_hits_reconstructed"][mask*mask_non_nan].values
            n_t = sd["number_unique_hits"][mask*mask_non_nan].values
            N = len(error_percentage_of_hits)
            error_percentage_of_hits_unique = (n_r/(n_t**2)*np.sqrt(n_t))**2+(1/n_t*np.sqrt(n_r))**2
            number_of_hits_var_unique.append(1/N*np.sqrt(np.sum(error_percentage_of_hits_unique)))
            total_reconstructed_50 = np.sum(percentage_of_hits_MC>0.50)
            total_reconstructed_75 = np.sum(percentage_of_hits_MC>0.75)
            eff.append(
                (total_reconstructed_cld_crit) / total_showers
            )

            eff_def1.append(
                (total_reconstructed_cld_crit_purity) / total_showers
            )
            eff_50.append(total_reconstructed_50/total_showers)
            eff_75.append(total_reconstructed_75/total_showers)
            energy_eff.append((bin_i1 + bin_i) / 2)
            size_energy_bin.append((bin_i1-bin_i)/2)
            total_showers_.append(total_showers)
            n_total = total_showers
            n_r = total_reconstructed_cld_crit
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors.append(error)

            n_r = total_reconstructed_cld_crit_purity
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors_def1.append(error)
        # print(
        #     "bin",
        #     bin_i1,
        #     bin_i,
        #     (total_showers - number_of_non_reconstructed_showers) / total_showers,
        #     total_showers,
        #     (total_showers - number_of_non_reconstructed_showers),
        #     error
        # )
    photons_dic = {}
    photons_dic["purity"]= purity
    photons_dic["number_of_hits_unique"]= number_of_hits_unique
    photons_dic["purity_var"]=purity_var
    # photons_dic["eff_50"] = eff_50
    # photons_dic["eff_75"] = eff_75
    photons_dic["eff"] = eff
    photons_dic["eff_def1"] = eff_def1
    photons_dic["number_of_hits"]=number_of_hits
    photons_dic["number_of_hits_var"]= number_of_hits_var
    photons_dic["number_of_hits_var_unique"] = number_of_hits_var_unique
    photons_dic["energy_eff"] = energy_eff
    photons_dic["total_showers_"] = total_showers_
    photons_dic["size_energy_bin"] = size_energy_bin
    photons_dic["errors"] = errors
    photons_dic["errors_def1"] = errors_def1
    return photons_dic


def calculate_eff_vertex(sd,photons_dic):
    # if log_scale:
   
    # else:
    bins = np.arange(0, 1000, 20)
    bins = np.array(bins)
    eff_v = []
    energy_v = []
    errors_v = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = (sd.R.values*1000) <= bin_i1
        mask_below = (sd.R.values*1000) > bin_i
        mask = mask_below * mask_above

        total_showers = len(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            # purity_calc = sd["e_pred_and_truth"][mask].values/sd["pred_showers_E"][mask].values
            # more_than_4_hits = sd["pred_showers_E"][mask].values>3
            # total_reconstructed_cld_crit =  np.sum((purity_calc>=0.75)*more_than_4_hits)
            total_reconstructed_cld_crit = np.sum(sd["e_pred_and_truth"][mask].values>3)
            # particle_purity = sd["e_pred_and_truth"][mask].values/sd["reco_showers_E"][mask].values
            # track_purity = sd["e_pred_and_truth"][mask].values/sd["pred_showers_E"][mask].values
            # more_than_4_hits = sd["pred_showers_E"][mask].values>3
            #total_reconstructed_cld_crit =  np.sum((purity_calc>=0.75)*more_than_4_hits)
            # total_reconstructed_cld_crit =  np.sum((particle_purity>=0.5)*(track_purity>=0.5)*more_than_4_hits)
            eff_v.append(
                (total_reconstructed_cld_crit) / total_showers
            )
            energy_v.append((bin_i1 + bin_i) / 2)
        
            n_total = total_showers
            n_r = total_reconstructed_cld_crit
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors_v.append(error)

    photons_dic["eff_v"]= eff_v
    photons_dic["energy_v"]=energy_v
    photons_dic["errors_v"] =errors_v

    return photons_dic



def calculate_eff_DeltaMC(sd, photons, dictlog_scale=False):
    
    bins = np.arange(0.001, 0.4,0.01)
    bins = np.exp(np.arange(np.log(1e-3), np.log(0.5), 0.5))
    eff = []
    energy_eff = []
    total_showers_ = []
    errors_delta_mc = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.delta_MC.values <= bin_i1
        mask_below = sd.delta_MC.values > bin_i
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_showers_E.values)[mask]
        )
        total_showers = len(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            # purity_calc = sd["e_pred_and_truth"][mask].values/sd["pred_showers_E"][mask].values
            # more_than_4_hits = sd["pred_showers_E"][mask].values>3
            # total_reconstructed_cld_crit =  np.sum((purity_calc>=0.75)*more_than_4_hits)
            total_reconstructed_cld_crit = np.sum(sd["e_pred_and_truth"][mask].values>3)
            # particle_purity = sd["e_pred_and_truth"][mask].values/sd["reco_showers_E"][mask].values
            # track_purity = sd["e_pred_and_truth"][mask].values/sd["pred_showers_E"][mask].values
            # more_than_4_hits = sd["pred_showers_E"][mask].values>3
            # total_reconstructed_cld_crit =  np.sum((particle_purity>=0.5)*(track_purity>=0.5)*more_than_4_hits)
            eff.append(
                (total_reconstructed_cld_crit) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)
            total_showers_.append(total_showers)
            n_total = total_showers
            n_r = total_reconstructed_cld_crit
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors_delta_mc.append(error)
    photons["eff_delta_MC"]= eff
    photons["delta_MC_values"]=energy_eff
    photons["errors_delta_mc"]=errors_delta_mc
    return photons



# reconstructable_ml[reconstructable_ml.true_showers_pt<0.7]
# sd = reconstructable_ml[reconstructable_ml.true_showers_pt<0.7]
# total_showers = len(sd.pred_showers_E.values)

# particle_purity = sd["e_pred_and_truth"].values/sd["reco_showers_E"].values
# track_purity = sd["e_pred_and_truth"].values/sd["pred_showers_E"].values
# more_than_4_hits = sd["pred_showers_E"].values>3
# mask1 = (particle_purity<0.5)+(track_purity<0.5)
# total_reconstructed_cld_crit =  np.sum((particle_purity<0.5)*(track_purity<0.5)*more_than_4_hits)
# total_reconstructed_cld_crit, total_showers, total_reconstructed_cld_crit/total_showers
# sd[mask1] 


def calculate_splits(sd_hgb_ct, sd_hgb):
    reconstructable_ct,reco_R_ct, reco_delta_plot, reconstructable_ct_all_p = make_cuts_ct(sd_hgb_ct)
    particle_purity = reconstructable_ct["e_pred_and_truth"].values/reconstructable_ct["reco_showers_E"].values
    track_purity = reconstructable_ct["e_pred_and_truth"].values/reconstructable_ct["pred_showers_E"].values
    good_ct = np.sum((particle_purity>=0.5)*(track_purity>=0.5))/len(reconstructable_ct)
    split_ct = np.sum((particle_purity<0.5)*(track_purity>=0.5))/len(reconstructable_ct)
    multiple_ct = np.sum((particle_purity>=0.5)*(track_purity<0.5))/len(reconstructable_ct)
    bad_ct = np.sum((particle_purity<0.5)*(track_purity<0.5))/len(reconstructable_ct)
    nan_ct = np.sum(np.isnan(particle_purity))/len(reconstructable_ct)

    reconstructable_ml,reco_R, reco_delta, reconstructable_all_p = make_cuts_ct(sd_hgb)
    particle_purity = reconstructable_ml["e_pred_and_truth"].values/reconstructable_ml["reco_showers_E"].values
    track_purity = reconstructable_ml["e_pred_and_truth"].values/reconstructable_ml["pred_showers_E"].values
    good_ml = np.sum((particle_purity>=0.5)*(track_purity>=0.5))/len(reconstructable_ml)
    split_ml = np.sum((particle_purity<0.5)*(track_purity>=0.5))/len(reconstructable_ml)
    multiple_ml = np.sum((particle_purity>=0.5)*(track_purity<0.5))/len(reconstructable_ml)
    bad_ml = np.sum((particle_purity<0.5)*(track_purity<0.5))/len(reconstructable_ml)
    nan_ml = np.sum(np.isnan(particle_purity))/len(reconstructable_ml)
    print(good_ct,split_ct, multiple_ct,bad_ct,nan_ct )
    print(good_ml,split_ml, multiple_ml,bad_ml,nan_ml )

    good_ml_idea = 0.9803275338323467
    split_ml_idea =0.00421367287489219
    multiple_ml_idea = 0.0019293183652404028
    bad_ml_idea = 4.8492741675701326e-05
    nan_ml_idea = 0.013480982185844969

    data = {
    'Algorithm': ['CT CLD', 'TGNN CLD','TGNN IDEA' ],
    'Good': [good_ct,good_ml,good_ml_idea ],
    'Split': [split_ct, split_ml,split_ml_idea],
    'Multiple': [multiple_ct, multiple_ml, multiple_ml_idea],
    'Bad': [bad_ct+nan_ct, bad_ml+nan_ml, bad_ml_idea+nan_ml_idea]
    }
    return data 