# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:22:41 2022

@author: Michael
"""

import requests
from bs4 import BeautifulSoup as bs
import json

playersite = requests.get('https://www.nba.com/players')
playersoup = bs(playersite.text, 'html.parser')
playerjson = json.loads(playersoup.html.body.contents[1].text)
playerlist = playerjson['props']['pageProps']['players']

