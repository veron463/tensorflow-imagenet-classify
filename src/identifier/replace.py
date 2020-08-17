
f = open('../../data/replace_all.csv', 'w')
filepath = '../../data/all.csv'
with open(filepath) as fp:
   line = fp.readline()

   cnt = 1
   while line:
       c_line = line.replace('/','"/', 1)
       c_line = c_line.replace('ep&eacute;e/smallsword','smallsword')
       c_line = c_line.replace('jpg', 'jpg"')
       c_line = c_line.replace('JPG', 'JPG"')
       print("{}".format(c_line))
       f.writelines(c_line)
       line = fp.readline()
       cnt += 1

# digits_re = r'/?var.+(\w.mail\b)'
# sample = '/usr/sbin/sendmail - 0 errors, 12 warnings'
# print(re.sub(digits_re, digits_re.replace('/', '"/'), sample))
# p = re.compile(r'/?var.+(\w.jpg\b)')
# str = p.match('TEST,/var/www/vhosts/oldswords.com/httpdocs/myhomeweb/mypictures/Part10/10261-s4c.jpg,American Academy '
#               'Air Force')
# print(str)
# df = pd.read_csv('../../data/all.csv', error_bad_lines=False, index_col=False)
