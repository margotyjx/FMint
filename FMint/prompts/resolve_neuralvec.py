def fill_caption_number(caption_template, identifier, params, format_str = '{:.3g}', finetune = True):
	if "lotka_volterra" in identifier:
		caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
								.replace('[0.002]', format_str.format(params[1])) \
								.replace('[0.003]', format_str.format(params[2])) \
								.replace('[0.004]', format_str.format(params[3])) \
								.replace('[0.005]', format_str.format(params[4]))
	elif "harmonic_oscillator" in identifier:
		caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
								.replace('[0.002]', format_str.format(params[1]))
	elif "vander_pol" in identifier:
		caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
								.replace('[0.002]', format_str.format(params[1]))
	elif "dampedharmonic_oscillator" in identifier:
		caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
								.replace('[0.002]', format_str.format(params[1])) \
								.replace('[0.003]', format_str.format(params[2]))
	elif "lorenz_attractor" in identifier:
		caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
								.replace('[0.002]', format_str.format(params[1])) \
								.replace('[0.003]', format_str.format(params[2])) \
								.replace('[0.004]', format_str.format(params[3]))
	else:
		if finetune:
			caption = caption_template
		else:	
			raise NotImplementedError
	
	return caption