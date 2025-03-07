"""
@file: params_manager.py
@breif: parameters manager
@author: Yang Haodong
@update: 2024.6.26
"""
import os
import yaml
import numpy as np

from PIL import Image

class ParamsManager:
	def __init__(self, config_file: str) -> None:
		# total parameters
		self.params = self.parse(config_file)

		if "strategy" not in self.params.keys():
			raise RuntimeError("parameters `strategy` should be configured.")
		if "planner" not in self.params["strategy"].keys():
			raise RuntimeError("parameters `planner` should be configured in `strategy`.")
		if "controller" not in self.params["strategy"].keys():
			raise RuntimeError("parameters `controller` should be configured in `strategy`.")

		# environment
		if "grid_map" in self.params.keys() and "file" in self.params["grid_map"].keys() \
			and self.params["grid_map"]["file"]:
			file = os.path.abspath(os.path.join(config_file, self.params["grid_map"]["file"]))
			img = Image.open(file)
			grid_map = np.array(img)[:, :, 0]
			grid_map[grid_map < 253] = 1
			grid_map[grid_map >= 253] = 0
			self.params["grid_map"].pop("file")
			self.params["grid_map"]["grid_map"] = grid_map
			self.params["grid_map"]["dimensions"] = [grid_map.shape[0], grid_map.shape[1]]

		extra_config = ["planner", "controller"]
		for obj in extra_config:
			if "config" in self.params["strategy"][obj].keys():
				extra_params = self.parse(os.path.abspath(os.path.join(
					config_file, self.params["strategy"][obj]["config"]
				)))
				if self.params["strategy"][obj]["name"] not in extra_params.keys():
					continue
				else:
					extra_params = extra_params[self.params["strategy"][obj]["name"]]
				self.params["strategy"][obj].pop("config")
				for k, v in extra_params.items():
					self.params["strategy"][obj][k] = v

	def parse(self, config_file: str) -> dict:
		params = dict()
		with open(config_file, 'r') as f:
			try:
				params = yaml.load(f, Loader=yaml.FullLoader)
			except yaml.YAMLError as exc:
				print(exc)
		return params
	
	def getParams(self) -> dict:
		return self.params