/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    EntropyBasedSplitCrit.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.C45.LiNing;

import weka.classifiers.myalgorithm.util.DecimalCalculate;
import weka.core.ContingencyTables;

/**
 * "Abstract" class for computing splitting criteria based on the entropy of a
 * class distribution.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10055 $
 */
public abstract class EntropyBasedSplitCrit extends SplitCriterion {

	/** for serialization */
	private static final long serialVersionUID = -2618691439791653056L;

	/**
	 * Help method for computing entropy.
	 */
	public final double lnFunc(double num) {

		// Constant hard coded for efficiency reasons
		if (num < 1e-6)
			return 0;
		else
			return ContingencyTables.lnFunc(num);
	}

	/**
	 * Computes entropy of distribution before splitting.
	 */
	public final double oldEnt(Distribution bags, double totalNoInst) {

		double returnValue = 0;
		int j;
//		int numClasses = bags.numClasses();
		for (j = 0; j < bags.numClasses(); j++) {
			returnValue += bags.perClass(j) * (totalNoInst - bags.perClass(j));
		}
		// 对信息熵做误差弥补
//		double result = DecimalCalculate.div(returnValue, totalNoInst) * numClasses;
//		if (bags.numBags() >= 10) {
//			returnValue = returnValue / totalNoInst;
//		} else {
			returnValue = (returnValue / totalNoInst) * bags.numClasses();
//		}

//		System.out.println("Entropy(S) ==> " + result);
		return returnValue;
	}

	/**
	 * Computes entropy of distribution before splitting.
	 */
	public final double oldEnt(Distribution bags) {

		double returnValue = 0;
		int j;

		for (j = 0; j < bags.numClasses(); j++) {
			returnValue += bags.perClass(j) * (bags.total() - bags.perClass(j));
		}
		// 对信息熵做误差弥补
		double result = DecimalCalculate.div(returnValue, bags.total()) * bags.numClasses();

//		System.out.println("Entropy(S) ==> " + result);
		return result;
	}

	/**
	 * Computes entropy of distribution after splitting.
	 */
	public final double newEnt(Distribution bags, double totalNoInst) {
		double returnValue = 0;
		int i, j;
		double sum;
		int numBags = bags.numBags();
		for (i = 0; i < numBags; i++) {
			sum = 0;
			if (bags.perBag(i) == 0) {
				continue;
			} else {
				for (j = 0; j < bags.numClasses(); j++) {
					sum += bags.perClassPerBag(i, j) * (bags.perBag(i) - bags.perClassPerBag(i, j));
				}
				
				returnValue += (sum / bags.perBag(i));
//				returnValue += DecimalCalculate.div(sum, bags.perBag(i));
			}
		}
//		System.out.println("Entropy(S,A) ==> " + returnValue);
		return returnValue;
	}

	/**
	 * Computes entropy of distribution after splitting.
	 */
	public final double newEnt(Distribution bags) {

		double returnValue = 0;
		int i, j;

		for (i = 0; i < bags.numBags(); i++) {
			double sum = 0;
			if (bags.perBag(i) == 0) {
				continue;
			} else {
				for (j = 0; j < bags.numClasses(); j++) {
					sum += bags.perClassPerBag(i, j) * (bags.perBag(i) - bags.perClassPerBag(i, j));
				}
				
				returnValue += DecimalCalculate.div(sum, bags.perBag(i));
			}
		}
//		System.out.println("Entropy(S,A) ==> " + returnValue);
		return returnValue;
	}

	/**
	 * Computes entropy after splitting without considering the class values.
	 */
	public final double splitEnt(Distribution bags) {

		double returnValue = 0;
		int i;

		for (i = 0; i < bags.numBags(); i++)
			returnValue = returnValue + lnFunc(bags.perBag(i));
		return (lnFunc(bags.total()) - returnValue) / ContingencyTables.log2;
	}
}
