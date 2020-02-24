import Converters
from Converters import genderConverter
from Converters import expConverter
from Converters import raceConverter
from Converters import grandCompanyConverter
import FFXIVInfo
from FFXIVInfo import usRealms
from FFXIVInfo import euRealms 
from FFXIVInfo import jpRealms 

class ClusterCharacter:
	def __init__(self,name,id=-1,race=-1,gender=-1,free_company=-1,realm=-1,grand_company=-1,legacy_player=-1,physicalMeeleExp=-1,physicalRangeExp=-1,tankExp=-1,magicRangeExp=-1,healerExp=-1,arcanistExp=-1,craftExp=-1,gatherExp=-1,minimalCertainPaidTime=-1,hildibrand=-1,preArr=-1,preHw=-1,physicalObject=-1,marriage=-1,beastTribesCompleted=-1):
		self.id = id
		self.name = name
		self.gender = gender
		self.free_company = free_company
		self.race = race
		self.realm = realm
		self.grand_company = grand_company
		self.legacy_player = legacy_player		
		self.physicalMeeleExp = physicalMeeleExp
		self.physicalRangeExp = physicalRangeExp
		self.tankExp = tankExp
		self.magicRangeExp = magicRangeExp
		self.healerExp = healerExp
		self.arcanistExp = arcanistExp
		self.craftExp = craftExp
		self.gatherExp = gatherExp		
		self.minimalCertainPaidTime = minimalCertainPaidTime
		self.hildibrand = hildibrand
		self.preArr = preArr
		self.preHw = preHw
		self.physicalObject = physicalObject
		self.beastTribesCompleted = beastTribesCompleted
		self.marriage = marriage

def getTuple(obj, items):
	values = []

	for item in items:
		values.append(getattr(obj, item))

	return tuple(values)

def getRaceNumber(raceString):
	raceNumber = raceConverter.get(raceString, -1)
	return raceNumber
	
def getGenderNumber(genderString):
	genderNumber = genderConverter.get(genderString, -1)
	return genderNumber

def getFreeCompanyBit(freeCompanyName):
	if (freeCompanyName == 'none'):
		return 0
	else:
		return 1


def getRealmNumber(realm):
	if(realm in jpRealms):
		return 0
	if(realm in usRealms):
		return 1
	if(realm in euRealms):
		return 2
	return -1


def getGrandCompanyNumber(grandCompanyString):
	grandCompanyNumber = grandCompanyConverter.get(grandCompanyString,-1)
	return grandCompanyNumber

def getMinimalCertainPaidTime(p30,p60,p90,p180,p270,p360,p450,p630,p960):
	if (p960==1):
		return 960/30
	if (p630==1):
		return 630/30
	if (p450==1):
		return 450/30
	if (p360==1):
		return 360/30
	if (p270==1):
		return 270/30
	if (p180==1):
		return 180/30
	if (p90==1):
		return 90/30
	if (p60==1):
		return 60/30
	if (p30==1):
		return 30/30
	return 0
	
def convertExp(lvl):
	exp = expConverter.get(lvl, "Invalid Level")
	return exp
			
	
def getClusterCharacters(characters):
	clusterCharacters = []
	for character in characters:
		id = character[0]
		name = character[1]
		race = getRaceNumber(character[2])
		gender = getGenderNumber(character[3])		
		free_company = getFreeCompanyBit(character[4])
		realm = getRealmNumber(character[5])
		grand_company = getGrandCompanyNumber(character[6])
		legacy_player = character[7]		
		physicalMeeleExp = convertExp(character[8])+convertExp(character[9])+convertExp(character[10])+convertExp(character[62])
		physicalRangeExp = convertExp(character[11])+convertExp(character[12])
		tankExp = convertExp(character[13])+convertExp(character[14])+convertExp(character[15])
		magicRangeExp = convertExp(character[16])+convertExp(character[63])
		healerExp = convertExp(character[17])+convertExp(character[18])
		arcanistExp = convertExp(character[19])
		craftExp = convertExp(character[20])+convertExp(character[21])+convertExp(character[22])+convertExp(character[23])+convertExp(character[24])+convertExp(character[25])+convertExp(character[26])+convertExp(character[27])
		gatherExp = convertExp(character[28])+convertExp(character[29])+convertExp(character[30])
		minimalCertainPaidTime = getMinimalCertainPaidTime(character[31],character[32],character[33],character[34],character[35],character[36],character[37],character[38],character[39])
		hildibrand = character[40]
		preArr = character[41]
		preHw = character[42]
		physicalObject = (character[43] + character[44] + character[45] + character[46] + character[47] + character[48] + character[49] + character[50] + character[51] + character[52])
		beastTribesCompleted = (character[53]+character[54]+character[55]+character[56]+character[57]+character[58]+character[59])
		marriage = character[60] or character[61]
		clusterCharacter = ClusterCharacter(name,id,race,gender,free_company,realm,grand_company,legacy_player,physicalMeeleExp,physicalRangeExp,tankExp,magicRangeExp,healerExp,arcanistExp,craftExp,gatherExp,minimalCertainPaidTime,hildibrand,preArr,preHw,physicalObject, marriage,beastTribesCompleted)
		clusterCharacters.append(clusterCharacter)
	return clusterCharacters
