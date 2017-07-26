sourceEncoding = "iso-8859-1"
targetEncoding = "utf-8"
source = open("movies_test_features.csv")
target = open("movies_test_features_utf8_1.csv", "w")

target.write(unicode(source.read(), sourceEncoding).encode(targetEncoding))