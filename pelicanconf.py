#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Anika Nawar'
SITENAME = 'Anika\'s blog'
SITEURL = 'https://nawarani.github.io'
PATH = 'content'
STATIC_PATHS = ['images']
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10
THEME = 'attila'
HOME_COVER = 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1352&q=80'
AUTHORS_BIO = {
  "zutrinken": {
    "name": "Anika",
    "cover": "https://arulrajnet.github.io/attila-demo/assets/images/avatar.png",
    "image": "https://arulrajnet.github.io/attila-demo/assets/images/avatar.png",
    "website": "anikanawar.com",
    "location": "Toronto",
    "bio": "Data scientist with a coffee addiction"
  }
}
SOCIAL = (('Twitter', 'https://twitter.com/anikanaw'),
          ('Github', 'https://github.com/nawarani'),
          ('Envelope','mailto:nawarani47@gmail.com'))
# COLOR_SCHEME_CSS = 'monokai.css'
# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False
